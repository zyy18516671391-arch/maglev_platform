import io
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.baselines import compare_with_newmark
from core.data import FEATURE_COLUMNS, generate_multimagnet_dataset, load_csv_dataset
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


def build_params(kc, use_structural_loss, lambda_physics, lambda_reconstruction):
    return {
        "m": 1.0,
        "g": 9.81,
        "c": 0.45,
        "k": 2.0,
        "epsilon": 0.05,
        "kc": kc,
        "physics_scale": 1e-3,
        "initial_weight": 1.0,
        "structural_weight": 0.1,
        "field_weight": 0.05,
        "field_div_scale": 1e-4,
        "field_flux_scale": 1e-3,
        "field_boundary_scale": 1e-3,
        "use_structural_loss": use_structural_loss,
        "lambda_physics": lambda_physics,
        "lambda_reconstruction": lambda_reconstruction,
        "lr": 0.001,
        "grad_clip": 5.0,
        "rho_a": 1.0,
        "beam_c": 0.08,
        "beam_k": 8.0,
        "slot_amp": 0.03,
        "slot_pitch": 0.6,
    }


def metric_dict(y_true, y_pred):
    true_np = y_true.detach().cpu().numpy().reshape(-1)
    pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    mse = mean_squared_error(true_np, pred_np)
    return {
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "MAE": mean_absolute_error(true_np, pred_np),
        "R2": r2_score(true_np, pred_np),
    }


def make_service(params, num_magnets, window, latent_dim):
    return MaglevService(
        params,
        num_magnets=num_magnets,
        feature_dim=len(FEATURE_COLUMNS),
        window=window,
        latent_dim=latent_dim,
    )


def render_history(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [
        ("data", "数据损失"),
        ("physics", "物理残差"),
        ("reconstruction", "重构损失"),
        ("initial", "初始条件"),
        ("structural", "结构残差"),
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


st.sidebar.header("数据与模型配置")
data_source = st.sidebar.radio("数据来源", ["增强仿真数据", "上传 CSV"], horizontal=False)
num_magnets = st.sidebar.slider("电磁铁数量", 2, 8, 4)
window = st.sidebar.slider("时间窗口长度", 4, 16, 8)
latent_dim = st.sidebar.slider("潜在维度", 8, 64, 32, step=8)
epochs = st.sidebar.slider("训练轮数", 20, 1000, 120, step=20)
noise = st.sidebar.slider("传感器噪声", 0.0, 0.10, 0.02, step=0.005)
sample_ratio = st.sidebar.slider("小样本比例", 0.25, 1.0, 0.75, step=0.05)
kc = st.sidebar.slider("相邻电磁铁耦合刚度 kc", 0.0, 5.0, 1.5, step=0.1)
lambda_physics = st.sidebar.slider("物理损失权重", 0.0, 1.0, 0.2, step=0.05)
lambda_reconstruction = st.sidebar.slider("重构损失权重", 0.0, 0.5, 0.05, step=0.01)
use_structural_loss = st.sidebar.checkbox("启用梁-齿槽结构动力学雏形", value=False)

params = build_params(kc, use_structural_loss, lambda_physics, lambda_reconstruction)

if data_source == "上传 CSV":
    uploaded_file = st.sidebar.file_uploader("上传工况 CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("请上传包含 t、I1..IN、gap1..gapN 的 CSV 文件。")
        st.stop()
    dataset = load_csv_dataset(uploaded_file, num_magnets=num_magnets, window=window)
else:
    dataset = generate_multimagnet_dataset(
        num_magnets=num_magnets,
        total_steps=240,
        window=window,
        noise_level=noise,
        sample_ratio=sample_ratio,
    )

st.caption(
    f"当前张量: seq={tuple(dataset.seq.shape)}, target={tuple(dataset.target_gap.shape)}, "
    f"features={dataset.feature_columns}"
)

tab_train, tab_compare, tab_deploy = st.tabs(["模型训练与评估", "算法对比", "离线推理与监测"])

with tab_train:
    mode = st.radio("训练模式", list(MODE_LABELS.keys()), format_func=lambda x: MODE_LABELS[x], horizontal=True)
    if st.button("启动训练", type="primary", key="train_one"):
        service = make_service(params, num_magnets, window, latent_dim)
        with st.spinner("正在训练多电磁铁 PINN 原型..."):
            history = service.train(
                dataset.seq,
                dataset.t,
                dataset.currents,
                dataset.target_gap,
                beam_disp=dataset.beam_disp,
                beam_field=dataset.beam_field,
                x_grid=dataset.x_grid,
                field_potential=dataset.field_potential,
                flux_density=dataset.flux_density,
                boundary=dataset.boundary,
                mode=mode,
                epochs=epochs,
            )

        full = service.predict_full(dataset.seq, dataset.t, dataset.currents)
        pred = full["gap"]
        metrics = metric_dict(dataset.target_gap, pred)
        cols = st.columns(4)
        cols[0].metric("MSE", f"{metrics['MSE']:.6f}")
        cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
        cols[2].metric("MAE", f"{metrics['MAE']:.4f}")
        cols[3].metric("R2", f"{metrics['R2'] * 100:.2f}%")

        c1, c2 = st.columns(2)
        with c1:
            render_history(history)
            render_latent(full, dataset.t)
        with c2:
            magnet_index = st.slider("查看电磁铁编号", 1, num_magnets, 1, key="single_magnet") - 1
            render_prediction(dataset.t, dataset.target_gap, pred, magnet_index)
            render_attention(full)

        buffer = io.BytesIO()
        torch.save(service.model.state_dict(), buffer)
        st.download_button(
            "下载当前模型权重",
            data=buffer.getvalue(),
            file_name=f"maglev_{mode}_{num_magnets}m.pth",
            mime="application/octet-stream",
        )

with tab_compare:
    if st.button("运行 Data-only / PINN / AE-PINN / Self-supervised 对比", type="primary", key="compare"):
        results = []
        preds = {}
        progress = st.progress(0)
        mode_order = ["data_only", "pinn", "ae_pinn", "self_supervised"]
        for idx, mode in enumerate(mode_order, start=1):
            service = make_service(params, num_magnets, window, latent_dim)
            history = service.train(
                dataset.seq,
                dataset.t,
                dataset.currents,
                dataset.target_gap,
                beam_disp=dataset.beam_disp,
                beam_field=dataset.beam_field,
                x_grid=dataset.x_grid,
                field_potential=dataset.field_potential,
                flux_density=dataset.flux_density,
                boundary=dataset.boundary,
                mode=mode,
                epochs=epochs,
            )
            pred = service.predict(dataset.seq, dataset.t, dataset.currents)
            metrics = metric_dict(dataset.target_gap, pred)
            metrics["模式"] = MODE_LABELS[mode]
            metrics["最终物理残差"] = history["physics"][-1]
            metrics["最终重构损失"] = history["reconstruction"][-1]
            metrics["最终梁 PDE 残差"] = history["structural"][-1]
            metrics["最终磁场残差"] = history["field"][-1]
            results.append(metrics)
            preds[mode] = pred
            progress.progress(idx / len(mode_order))

        st.dataframe(pd.DataFrame(results).set_index("模式"), use_container_width=True)
        if "ae_pinn" in preds:
            baseline_info = compare_with_newmark(preds["ae_pinn"], dataset.target_gap, dataset.t, dataset.currents, params)
            st.subheader("传统 Newmark-beta 基准")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "方法": "Newmark-beta",
                            "MSE": baseline_info["newmark_mse"],
                            "MAE": baseline_info["newmark_mae"],
                            "耗时 ms": baseline_info["elapsed_ms"],
                        },
                        {
                            "方法": "AE-PINN",
                            "MSE": baseline_info["model_mse"],
                            "MAE": metric_dict(dataset.target_gap, preds["ae_pinn"])["MAE"],
                            "耗时 ms": None,
                        },
                    ]
                ).set_index("方法"),
                use_container_width=True,
            )
        fig, ax = plt.subplots(figsize=(10, 4))
        t_np = dataset.t.detach().cpu().numpy().reshape(-1)
        ax.plot(t_np, dataset.target_gap[:, 0].detach().cpu().numpy(), "k-", alpha=0.25, lw=3, label="参考间隙")
        for mode, pred in preds.items():
            ax.plot(t_np, pred[:, 0].detach().cpu().numpy(), label=MODE_LABELS[mode])
        ax.set_title("三种训练模式的一号电磁铁响应对比")
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("间隙 / m")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)

with tab_deploy:
    st.subheader("离线模型加载")
    model_file = st.file_uploader("上传 .pth 权重", type=["pth"], key="model_upload")
    test_file = st.file_uploader("上传待测 CSV", type=["csv"], key="test_upload")
    if model_file and test_file and st.button("执行离线推理", type="primary"):
        service = make_service(params, num_magnets, window, latent_dim)
        try:
            service.model.load_state_dict(torch.load(io.BytesIO(model_file.read()), map_location="cpu"))
            test_data = load_csv_dataset(test_file, num_magnets=num_magnets, window=window)
            start = time.time()
            pred = service.predict(test_data.seq, test_data.t, test_data.currents)
            elapsed = (time.time() - start) * 1000
            metrics = metric_dict(test_data.target_gap, pred)
            c1, c2, c3 = st.columns(3)
            c1.metric("R2", f"{metrics['R2'] * 100:.2f}%")
            c2.metric("MAE", f"{metrics['MAE']:.4f}")
            c3.metric("推理耗时", f"{elapsed:.2f} ms")
            render_prediction(test_data.t, test_data.target_gap, pred, 0)
        except Exception as exc:
            st.error(f"离线推理失败：{exc}")
    else:
        st.info("使用相同电磁铁数量、窗口长度和潜在维度训练出的权重进行离线推理。")
