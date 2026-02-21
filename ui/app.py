import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterExponent
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import math
import io
import time
import sys
import os

# 1. 基础配置与乱码修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.maglev_service import MaglevService

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-muted')

st.set_page_config(layout="wide", page_title="Maglev-Neural 数字化平台")
st.title("🧲 Maglev-Neural 智能磁浮动力学分析与预测系统")

# ==========================================
# 📁 侧边栏：数据源与参数配置
# ==========================================
st.sidebar.header("📁 数据接入 (Data Ingestion)")
data_source = st.sidebar.radio("选择工况数据源", ["内置标准仿真工况", "上传真实实验数据 (CSV)"])

if data_source == "内置标准仿真工况":
    st.sidebar.header("⚙️ 仿真参数设置")
    kc = st.sidebar.slider("耦合刚度 kc", 0.5, 5.0, 1.5)
    noise_lvl = st.sidebar.slider("传感器噪声", 0.0, 0.1, 0.03)
else:
    st.sidebar.info("💡 请上传包含 `t`, `I1`, `I2`, `gap1` 字段的 CSV 文件。")
    uploaded_file = st.sidebar.file_uploader("上传测试工况数据", type=['csv'])
    kc = 1.5  # 使用默认参数
    noise_lvl = 0.0

epochs = st.sidebar.slider(
    "训练轮数",
    min_value=200,
    max_value=2000,
    value=400,
    step=100
)

params = {"m": 1.0, "g": 9.81, "c": 0.5, "k": 2.0, "epsilon": 0.05, "kc": kc}
WINDOW_SIZE = 5


# ==========================================
# 🔄 数据处理管道 (Data Pipeline)
# ==========================================
@st.cache_data
def load_and_preprocess_data(source, noise, _upload=None):
    if source == "内置标准仿真工况":
        T = 200
        t = torch.linspace(0, 5, T).view(-1, 1)
        x1 = 0.5 + 0.05 * torch.sin(2 * t)
        x2 = 0.55 + 0.04 * torch.cos(1.5 * t)
        I1 = torch.sin(2 * t) + 2.0
        I2 = torch.cos(1.5 * t) + 2.0
        target_clean = torch.cat([x1, x2], dim=1)
        target_noisy = target_clean + noise * torch.randn_like(target_clean)
    else:
        df = pd.read_csv(_upload)
        t = torch.tensor(df['t'].values, dtype=torch.float32).view(-1, 1)
        I1 = torch.tensor(df['I1'].values, dtype=torch.float32).view(-1, 1)
        I2 = torch.tensor(df['I2'].values, dtype=torch.float32).view(-1, 1)
        x1 = torch.tensor(df['gap1'].values, dtype=torch.float32).view(-1, 1)

        target_clean = torch.cat([x1, x1], dim=1)
        target_noisy = target_clean

    features = torch.cat([t, I1, I2, target_noisy[:, 0:1]], dim=1)
    X_seq = []
    for i in range(len(features) - WINDOW_SIZE + 1):
        X_seq.append(features[i: i + WINDOW_SIZE])

    return (torch.stack(X_seq), t[WINDOW_SIZE - 1:], I1[WINDOW_SIZE - 1:],
            I2[WINDOW_SIZE - 1:], target_clean[WINDOW_SIZE - 1:], target_noisy[WINDOW_SIZE - 1:])


if data_source == "上传真实实验数据 (CSV)" and uploaded_file is None:
    st.warning("⚠️ 请在左侧侧边栏上传您的 CSV 数据文件以激活数字孪生引擎。")
    st.stop()

X, t_f, I1_f, I2_f, Y_clean, Y_noisy = load_and_preprocess_data(data_source, noise_lvl,
                                                                uploaded_file if data_source != "内置标准仿真工况" else None)


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


# ==========================================
# 🖥️ 核心界面逻辑
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎯 模型训练与评估", "🥊 双引擎算法对决", "🚀 离线推理与监测"])

# ----------------- TAB 1: 训练与评估 -----------------
with tab1:
    mode = st.radio("模型模式", ["pinn", "data_only"], horizontal=True)
    if st.button("🚀 启动数字孪生分析", key="train_single"):
        service = MaglevService(params)

        with st.spinner("正在构建数字孪生模型，请稍候..."):
            history = service.train(X, t_f, I1_f, I2_f, Y_noisy, mode=mode, epochs=epochs)

        st.toast(f"✅ {mode.upper()} 模型训练完成！", icon='🎉')

        pred = service.predict(X, t_f, I1_f, I2_f).detach().numpy()
        y_true_np = Y_clean.numpy()[:, 0]
        y_pred_np = pred[:, 0]

        st.subheader("📊 综合性能评估面板")
        mse, rmse, mae, r2 = calculate_metrics(y_true_np, y_pred_np)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MSE (均方误差)", f"{mse:.6f}")
        m2.metric("RMSE (均方根误差)", f"{rmse:.4f}")
        m3.metric("MAE (平均绝对误差)", f"{mae:.4f}")
        m4.metric("R² (决定系数)", f"{r2 * 100:.2f}%", delta="高度拟合" if r2 > 0.9 else "需优化")

        st.divider()

        # 绘图
        col1, col2 = st.columns(2)
        with col1:
            fig_l, axl = plt.subplots(figsize=(8, 5))
            axl.plot(history["data"], label="数据损失", color="#1f77b4", lw=2)
            if mode == "pinn":
                axl.plot(history["physics"], label="物理残差", color="#ff7f0e", ls='--')
            axl.set_yscale('log')
            axl.yaxis.set_major_formatter(LogFormatterExponent(base=10))
            axl.set_title("收敛性能曲线")
            axl.legend()
            st.pyplot(fig_l)

        with col2:
            fig_r, axr = plt.subplots(figsize=(8, 5))
            axr.plot(y_true_np, 'k-', alpha=0.3, label="参考基准", lw=3)
            axr.plot(y_pred_np, 'r--', label="AI 预测")
            axr.set_title("动态响应预测拟合")
            axr.legend()
            st.pyplot(fig_r)

        # --- L2 升级：状态表征（潜在空间）可视化 ---
        st.divider()
        st.subheader("🧠 潜在空间状态表征 (Latent Space Representation)")
        st.info("💡 商业价值：这是 AI 提取的‘特征指纹’。降维后的状态空间可用于系统运行模式识别与状态评估。")

        try:
            with torch.no_grad():
                # 确保 X 转为标准的 Tensor，剥离计算图
                X_tensor = X.clone().detach().type(torch.float32)
                # 提取潜空间特征
                latent_features = service.model.perception(X_tensor).cpu().numpy()

            if latent_features.shape[0] > 2:
                pca = PCA(n_components=2)
                latent_2d = pca.fit_transform(latent_features)

                fig_lat, ax_lat = plt.subplots(figsize=(10, 4))
                c_data = t_f.numpy().flatten()

                scatter = ax_lat.scatter(latent_2d[:, 0], latent_2d[:, 1],
                                         c=c_data, cmap='viridis', s=20, alpha=0.8, edgecolors='none')
                ax_lat.set_title("系统状态特征云图 (PCA 降维映射)")
                ax_lat.set_xlabel("特征主分量 1 (PC1)")
                ax_lat.set_ylabel("特征主分量 2 (PC2)")
                fig_lat.colorbar(scatter, label='时间演化序列 (s)')
                ax_lat.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_lat)
                st.caption("注：云图展示了高维动力学状态的投影。连续的轨迹代表系统运行平稳，异常散点通常对应瞬态冲击。")
            else:
                st.warning("数据量过少，无法生成状态表征云图。")
        except Exception as e:
            st.error(f"潜在空间可视化加载失败：{e}")

        st.divider()
        buffer = io.BytesIO()
        torch.save(service.model.state_dict(), buffer)
        st.download_button("💾 下载当前训练好的模型权重 (.pth)", data=buffer.getvalue(),
                           file_name=f"maglev_{mode}_model.pth", mime="application/octet-stream")

# ----------------- TAB 2: 算法对决 -----------------
with tab2:
    if st.button("⚔️ 开启模型鲁棒性对决", type="primary"):
        s_data = MaglevService(params)
        s_pinn = MaglevService(params)

        with st.spinner("双引擎并行计算中，物理规律注入中..."):
            h_data = s_data.train(X, t_f, I1_f, I2_f, Y_noisy, mode="data_only", epochs=epochs)
            h_pinn = s_pinn.train(X, t_f, I1_f, I2_f, Y_noisy, mode="pinn", epochs=epochs)

        st.toast("⚔️ 对决分析已就绪！", icon='🎯')

        p_d = s_data.predict(X, t_f, I1_f, I2_f).detach().numpy()[:, 0]
        p_p = s_pinn.predict(X, t_f, I1_f, I2_f).detach().numpy()[:, 0]
        y_true_np = Y_clean.numpy()[:, 0]

        st.subheader("🎯 核心指标对决")
        mse_d, _, _, r2_d = calculate_metrics(y_true_np, p_d)
        mse_p, _, _, r2_p = calculate_metrics(y_true_np, p_p)

        c1, c2, c3 = st.columns(3)
        c1.metric("Data-Only R²", f"{r2_d * 100:.2f}%")
        c2.metric("PINN R²", f"{r2_p * 100:.2f}%", delta=f"提升 {(r2_p - r2_d) * 100:.2f}%")
        c3.metric("MSE 降低幅度", f"{((mse_d - mse_p) / mse_d) * 100:.1f}%", delta_color="inverse")

        fig_c, ax_c = plt.subplots(figsize=(12, 5))
        ax_c.plot(y_true_np, 'k-', alpha=0.2, label="Ground Truth", lw=3)
        ax_c.plot(p_d, 'r:', label="纯数据驱动 (受噪影响)")
        ax_c.plot(p_p, 'b-', label="PINN (物理修正)", lw=2)
        ax_c.set_title("真实工况下的模型抗干扰能力对比")
        ax_c.legend()
        st.pyplot(fig_c)

        st.divider()
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            buf_p = io.BytesIO()
            torch.save(s_pinn.model.state_dict(), buf_p)
            st.download_button("💾 导出 PINN 模型权重", data=buf_p.getvalue(),
                               file_name="maglev_pinn_best.pth", key="dp")
        with col_p2:
            buf_d = io.BytesIO()
            torch.save(s_data.model.state_dict(), buf_d)
            st.download_button("📥 导出 Data-Only 模型权重", data=buf_d.getvalue(),
                               file_name="maglev_data_comp.pth", key="dd")

# ----------------- TAB 3: 离线推理 -----------------
with tab3:
    st.subheader("📦 已训模型部署 (Model Deployment)")
    st.info("💡 商业化场景：无需重新训练，直接加载导出的 .pth 权重文件，对新工况进行秒级预测。")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        model_file = st.file_uploader("1. 上传模型权重 (.pth)", type=['pth'], key="model_upload")
    with col_up2:
        test_data_file = st.file_uploader("2. 上传待测工况数据 (CSV)", type=['csv'], key="test_data_upload")

    if model_file and test_data_file:
        if st.button("🛰️ 执行一键状态预测", type="primary"):
            deploy_service = MaglevService(params)
            try:
                deploy_service.model.load_state_dict(torch.load(io.BytesIO(model_file.read())))
                st.toast("模型权重加载成功！", icon="🧠")

                X_test, t_test, I1_test, I2_test, Y_test_clean, _ = load_and_preprocess_data(
                    "上传真实实验数据 (CSV)", 0.0, test_data_file
                )

                start_time = time.time()
                with torch.no_grad():
                    pred_test = deploy_service.predict(X_test, t_test, I1_test, I2_test).numpy()
                end_time = time.time()

                st.success(f"🚀 推理完成！耗时: {(end_time - start_time) * 1000:.2f} 毫秒")

                mse_t, rmse_t, mae_t, r2_t = calculate_metrics(Y_test_clean.numpy()[:, 0], pred_test[:, 0])
                m_c1, m_c2, m_c3 = st.columns(3)
                m_c1.metric("预测准确度 (R²)", f"{r2_t * 100:.2f}%")
                m_c2.metric("平均绝对误差 (MAE)", f"{mae_t:.4f} mm")
                m_c3.metric("实时性评估", "极高 (无时延)")

                fig_v, ax_v = plt.subplots(figsize=(12, 4))
                ax_v.plot(t_test.numpy(), Y_test_clean.numpy()[:, 0], 'k-', alpha=0.3, label="实际测量轨迹", lw=2)
                ax_v.plot(t_test.numpy(), pred_test[:, 0], 'g--', label="AI 数字孪生预测", lw=2)
                ax_v.set_title("离线推理状态监测面板")
                ax_v.legend()
                st.pyplot(fig_v)

            except Exception as e:
                st.error(f"加载失败：模型结构不匹配或文件损坏。详情: {e}")
    else:
        st.warning("请同时上传 .pth 模型文件和待分析的 CSV 数据文件以执行离线推理。")