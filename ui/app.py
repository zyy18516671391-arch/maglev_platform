import io
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.baselines import compare_with_newmark
from core.data import (
    FEATURE_COLUMNS,
    generate_fem_like_validation_dataset,
    generate_multimagnet_dataset,
    load_csv_dataset,
    load_fem_validation_csv,
)
from core.evaluator import evaluate_splits, regression_metrics
from core.experiments import flatten_split_metrics, save_experiment_record
from core.field import compute_field_quantities
from services.maglev_service import MaglevService


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-muted")

st.set_page_config(layout="wide", page_title="磁浮电磁耦合动力学 PINN 平台")
st.title("磁浮电磁耦合动力学 PINN 原型平台")

MODE_LABELS = {
    "data_only": "Data-only 基线",
    "pinn": "耦合 PINN",
    "ae_pinn": "自编码器 PINN",
    "self_supervised": "Self-supervised 物理自学习",
}

FIELD_MODE_LABELS = {
    "engineering": "工程力",
    "legacy_gradient": "旧梯度代理力",
    "maxwell_stress": "Maxwell 应力近似力",
    "blended": "工程力 + Maxwell 混合",
}


def build_params(kc, use_structural_loss, lambda_physics, lambda_reconstruction, field_force_model, field_force_blend):
    return {
        "m": 1.0,
        "g": 9.81,
        "c": 0.45,
        "k": 2.0,
        "epsilon": 0.05,
        "min_gap": 0.02,
        "force_limit": 100.0,
        "kc": kc,
        "physics_scale": 1e-3,
        "initial_weight": 1.0,
        "structural_weight": 0.1,
        "field_weight": 0.05,
        "field_div_scale": 1e-4,
        "field_flux_scale": 1e-3,
        "field_boundary_scale": 1e-3,
        "field_force_model": field_force_model,
        "field_force_blend": field_force_blend,
        "field_force_scale": 0.1,
        "mu0": 4 * 3.141592653589793 * 1e-7,
        "pole_area": 0.01,
        "force_normal": 1.0,
        "maxwell_force_scale": 1e-5,
        "use_structural_loss": use_structural_loss,
        "lambda_physics": lambda_physics,
        "lambda_reconstruction": lambda_reconstruction,
        "lr": 0.001,
        "grad_clip": 5.0,
        "rho_a": 1.0,
        "beam_c": 0.08,
        "beam_ei": 8.0,
        "functional_ei_scale": 0.75,
        "joint_ei_scale": 0.55,
        "slot_amp": 0.03,
        "slot_harmonics": 3,
        "slot_pitch": 0.6,
    }


def make_service(params, num_magnets, window, latent_dim):
    return MaglevService(
        params,
        num_magnets=num_magnets,
        feature_dim=len(FEATURE_COLUMNS),
        window=window,
        latent_dim=latent_dim,
    )


def train_service(service, dataset, mode, epochs):
    return service.train(
        dataset.seq,
        dataset.t,
        dataset.currents,
        dataset.target_gap,
        beam_disp=dataset.beam_disp,
        beam_field=dataset.beam_field,
        x_grid=dataset.x_grid,
        magnet_positions=dataset.magnet_positions,
        field_potential=dataset.field_potential,
        flux_density=dataset.flux_density,
        boundary=dataset.boundary,
        train_indices=dataset.split_indices["train"],
        mode=mode,
        epochs=epochs,
    )


def render_history(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [
        ("data", "数据损失"),
        ("physics", "物理残差"),
        ("reconstruction", "重构损失"),
        ("initial", "初始/边界条件"),
        ("structural", "梁 PDE 残差"),
        ("field", "磁场残差"),
    ]:
        values = history.get(key, [])
        if values and max(values) > 0:
            ax.plot(values, label=label)
    ax.set_yscale("log")
    ax.set_title("训练收敛曲线")
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def render_prediction(t, target, pred, magnet_index=0):
    fig, ax = plt.subplots(figsize=(8, 4))
    t_np = t.detach().cpu().numpy().reshape(-1)
    ax.plot(t_np, target[:, magnet_index].detach().cpu().numpy(), "k-", alpha=0.35, lw=3, label="参考间隙")
    ax.plot(t_np, pred[:, magnet_index].detach().cpu().numpy(), "r--", lw=2, label="模型预测")
    ax.set_title(f"电磁铁 {magnet_index + 1} 动态间隙响应")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("间隙 / m")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)


def render_latent(full_output, t):
    latent = full_output["latent"].detach().cpu().numpy()
    if latent.shape[0] < 3:
        st.info("样本数不足，暂不显示潜在空间。")
        return
    latent_2d = PCA(n_components=2).fit_transform(latent)
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=t.detach().cpu().numpy().reshape(-1), cmap="viridis", s=22)
    ax.set_title("低维状态表征 PCA 投影")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(scatter, label="时间 / s")
    st.pyplot(fig)


def render_attention(full_output):
    c1, c2 = st.columns(2)
    temporal = full_output.get("temporal_attention", full_output.get("attention"))
    spatial = full_output.get("spatial_attention")
    if temporal is not None:
        attn = temporal.detach().cpu().numpy().mean(axis=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(attn, cmap="magma")
        ax.set_title("时间自注意力权重")
        ax.set_xlabel("Key 时间步")
        ax.set_ylabel("Query 时间步")
        fig.colorbar(im, ax=ax)
        c1.pyplot(fig)
    if spatial is not None:
        attn = spatial.detach().cpu().numpy().mean(axis=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(attn, cmap="viridis")
        ax.set_title("电磁铁空间注意力权重")
        ax.set_xlabel("Key 电磁铁")
        ax.set_ylabel("Query 电磁铁")
        fig.colorbar(im, ax=ax)
        c2.pyplot(fig)


def render_force_comparison(dataset, params, magnet_index=0):
    engineering = params["k"] * dataset.currents ** 2 / (torch.clamp(dataset.target_gap, min=1e-4) + params["epsilon"]) ** 2
    field = compute_field_quantities(
        dataset.target_gap,
        dataset.currents,
        dataset.magnet_positions,
        params,
        field_potential=dataset.field_potential,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    t_np = dataset.t.detach().cpu().numpy().reshape(-1)
    ax.plot(t_np, engineering[:, magnet_index].detach().cpu().numpy(), label="工程力")
    ax.plot(t_np, field["maxwell_force"][:, magnet_index].detach().cpu().numpy(), label="Maxwell 应力近似力")
    if dataset.force_ref is not None:
        ax.plot(t_np, dataset.force_ref[:, magnet_index].detach().cpu().numpy(), "k--", alpha=0.65, label="force_ref")
    ax.set_title(f"电磁铁 {magnet_index + 1} 力模型对比")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("力 / proxy unit")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)


def split_metrics_dataframe(metrics):
    rows = []
    for split_name, values in metrics.items():
        row = {"数据集": split_name}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows).set_index("数据集")


st.sidebar.header("数据与模型配置")
data_source = st.sidebar.radio(
    "数据来源",
    ["增强仿真数据", "FEM-like 半真实验证集", "上传宽表 CSV", "上传 FEM 长表 CSV"],
    horizontal=False,
)
num_magnets = st.sidebar.slider("电磁铁数量", 2, 8, 4)
beam_points = st.sidebar.slider("梁 PDE 配点数", 9, 65, 33, step=4)
window = st.sidebar.slider("时间窗口长度", 4, 16, 8)
latent_dim = st.sidebar.slider("潜在维度", 8, 64, 32, step=8)
epochs = st.sidebar.slider("训练轮数", 20, 1000, 120, step=20)
noise = st.sidebar.slider("传感器噪声", 0.0, 0.10, 0.02, step=0.005)
sample_ratio = st.sidebar.slider("小样本比例", 0.25, 1.0, 0.75, step=0.05)
kc = st.sidebar.slider("相邻电磁铁耦合刚度 kc", 0.0, 5.0, 1.5, step=0.1)
lambda_physics = st.sidebar.slider("物理损失权重", 0.0, 1.0, 0.2, step=0.05)
lambda_reconstruction = st.sidebar.slider("重构损失权重", 0.0, 0.5, 0.05, step=0.01)
field_force_model = st.sidebar.selectbox(
    "场力模型",
    list(FIELD_MODE_LABELS.keys()),
    index=3,
    format_func=lambda key: FIELD_MODE_LABELS[key],
)
field_force_blend = st.sidebar.slider("Maxwell 力混合系数", 0.0, 1.0, 0.3, step=0.05)
use_structural_loss = st.sidebar.checkbox("启用梁/齿槽结构动力学残差", value=False)

params = build_params(kc, use_structural_loss, lambda_physics, lambda_reconstruction, field_force_model, field_force_blend)

if data_source == "上传宽表 CSV":
    uploaded_file = st.sidebar.file_uploader("上传工况 CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("请上传包含 t、I1..IN、gap1..gapN 的宽表 CSV 文件。")
        st.stop()
    dataset = load_csv_dataset(uploaded_file, num_magnets=num_magnets, window=window, beam_points=beam_points)
elif data_source == "上传 FEM 长表 CSV":
    uploaded_file = st.sidebar.file_uploader("上传 FEM/COMSOL/ANSYS 长表 CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("请上传包含 case_id,time,magnet_id,x,current,gap,flux_density,field_potential,force_ref,beam_disp 的 CSV 文件。")
        st.stop()
    dataset = load_fem_validation_csv(uploaded_file, window=window, beam_points=beam_points)
    num_magnets = dataset.currents.shape[1]
elif data_source == "FEM-like 半真实验证集":
    dataset = generate_fem_like_validation_dataset(
        num_magnets=num_magnets,
        steps_per_case=80,
        window=window,
        beam_points=beam_points,
    )
else:
    dataset = generate_multimagnet_dataset(
        num_magnets=num_magnets,
        total_steps=240,
        window=window,
        noise_level=noise,
        sample_ratio=sample_ratio,
        beam_points=beam_points,
    )

split_sizes = {name: len(indices) for name, indices in dataset.split_indices.items()}
st.caption(
    f"数据源={dataset.source_type}, seq={tuple(dataset.seq.shape)}, target={tuple(dataset.target_gap.shape)}, "
    f"x_grid={tuple(dataset.x_grid.shape)}, cases={len(dataset.case_ids.unique())}, split={split_sizes}"
)

tab_train, tab_compare, tab_deploy = st.tabs(["模型训练与评估", "算法对比", "离线推理与监测"])

with tab_train:
    mode = st.radio("训练模式", list(MODE_LABELS.keys()), format_func=lambda x: MODE_LABELS[x], horizontal=True)
    if st.button("启动训练", type="primary", key="train_one"):
        service = make_service(params, dataset.currents.shape[1], window, latent_dim)
        with st.spinner("正在训练多电磁铁 PINN 原型..."):
            history = train_service(service, dataset, mode, epochs)

        full = service.predict_full(dataset.seq, dataset.t, dataset.currents)
        pred = full["gap"]
        split_metrics = evaluate_splits(service.model, dataset)
        full_metrics = regression_metrics(dataset.target_gap, pred)
        cols = st.columns(4)
        cols[0].metric("全量 MSE", f"{full_metrics['MSE']:.6f}")
        cols[1].metric("全量 RMSE", f"{full_metrics['RMSE']:.4f}")
        cols[2].metric("全量 MAE", f"{full_metrics['MAE']:.4f}")
        cols[3].metric("全量 R2", f"{full_metrics['R2'] * 100:.2f}%")
        st.dataframe(split_metrics_dataframe(split_metrics), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            render_history(history)
            render_latent(full, dataset.t)
            render_force_comparison(dataset, params, 0)
        with c2:
            magnet_index = st.slider("查看电磁铁编号", 1, dataset.currents.shape[1], 1, key="single_magnet") - 1
            render_prediction(dataset.t, dataset.target_gap, pred, magnet_index)
            render_force_comparison(dataset, params, magnet_index)
            render_attention(full)

        buffer = io.BytesIO()
        torch.save(service.model.state_dict(), buffer)
        st.download_button(
            "下载当前模型权重",
            data=buffer.getvalue(),
            file_name=f"maglev_{mode}_{dataset.currents.shape[1]}m.pth",
            mime="application/octet-stream",
        )

with tab_compare:
    if st.button("运行四种模式对比", type="primary", key="compare"):
        results = []
        preds = {}
        records = []
        progress = st.progress(0)
        mode_order = ["data_only", "pinn", "ae_pinn", "self_supervised"]
        for idx, mode_name in enumerate(mode_order, start=1):
            service = make_service(params, dataset.currents.shape[1], window, latent_dim)
            start = time.perf_counter()
            history = train_service(service, dataset, mode_name, epochs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            pred = service.predict(dataset.seq, dataset.t, dataset.currents)
            split_metrics = evaluate_splits(service.model, dataset)
            row = {"模式": MODE_LABELS[mode_name]}
            row.update(flatten_split_metrics(split_metrics))
            if dataset.force_ref is not None:
                row["force_ref_MAE"] = torch.mean(torch.abs(dataset.force_ref - params["k"] * dataset.currents ** 2 / (dataset.target_gap + params["epsilon"]) ** 2)).item()
            row["最终总损失"] = history["total"][-1]
            row["最终物理残差"] = history["physics"][-1]
            row["最终重构损失"] = history["reconstruction"][-1]
            row["最终梁 PDE 残差"] = history["structural"][-1]
            row["最终磁场残差"] = history["field"][-1]
            row["耗时 ms"] = elapsed_ms
            results.append(row)
            preds[mode_name] = pred
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode_name,
                "params": params,
                "source_type": dataset.source_type,
                "elapsed_ms": elapsed_ms,
                "final_loss": history["total"][-1],
                "final_components": {
                    "data": history["data"][-1],
                    "physics": history["physics"][-1],
                    "reconstruction": history["reconstruction"][-1],
                    "structural": history["structural"][-1],
                    "field": history["field"][-1],
                },
                "split_metrics": split_metrics,
            }
            records.append(save_experiment_record(record))
            progress.progress(idx / len(mode_order))

        st.dataframe(pd.DataFrame(results).set_index("模式"), use_container_width=True)
        st.success("实验记录已保存：" + "；".join(path_info["json"] for path_info in records))
        if "ae_pinn" in preds:
            baseline_info = compare_with_newmark(preds["ae_pinn"], dataset.target_gap, dataset.t, dataset.currents, params)
            baseline_metrics = regression_metrics(dataset.target_gap, baseline_info["baseline"])
            st.subheader("非线性 Newmark-beta 基准")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"方法": baseline_info["method"], **baseline_metrics, "耗时 ms": baseline_info["elapsed_ms"]},
                        {"方法": "AE-PINN", **regression_metrics(dataset.target_gap, preds["ae_pinn"]), "耗时 ms": None},
                    ]
                ).set_index("方法"),
                use_container_width=True,
            )
        fig, ax = plt.subplots(figsize=(10, 4))
        t_np = dataset.t.detach().cpu().numpy().reshape(-1)
        ax.plot(t_np, dataset.target_gap[:, 0].detach().cpu().numpy(), "k-", alpha=0.25, lw=3, label="参考间隙")
        for mode_name, pred in preds.items():
            ax.plot(t_np, pred[:, 0].detach().cpu().numpy(), label=MODE_LABELS[mode_name])
        ax.set_title("不同训练模式的一号电磁铁响应对比")
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("间隙 / m")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)

with tab_deploy:
    st.subheader("离线模型加载")
    model_file = st.file_uploader("上传 .pth 权重", type=["pth"], key="model_upload")
    test_file = st.file_uploader("上传待测宽表 CSV", type=["csv"], key="test_upload")
    if model_file and test_file and st.button("执行离线推理", type="primary"):
        service = make_service(params, dataset.currents.shape[1], window, latent_dim)
        try:
            service.model.load_state_dict(torch.load(io.BytesIO(model_file.read()), map_location="cpu"))
            test_data = load_csv_dataset(test_file, num_magnets=dataset.currents.shape[1], window=window, beam_points=beam_points)
            start = time.time()
            pred = service.predict(test_data.seq, test_data.t, test_data.currents)
            elapsed = (time.time() - start) * 1000
            metrics = regression_metrics(test_data.target_gap, pred)
            c1, c2, c3 = st.columns(3)
            c1.metric("R2", f"{metrics['R2'] * 100:.2f}%")
            c2.metric("MAE", f"{metrics['MAE']:.4f}")
            c3.metric("推理耗时", f"{elapsed:.2f} ms")
            render_prediction(test_data.t, test_data.target_gap, pred, 0)
        except Exception as exc:
            st.error(f"离线推理失败：{exc}")
    else:
        st.info("请使用相同电磁铁数量、窗口长度、梁配点数和潜在维度训练出的权重进行离线推理。")
