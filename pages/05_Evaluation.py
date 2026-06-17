from __future__ import annotations

import time
import pandas as pd
import streamlit as st
from src import eval_utils, ui_utils
from src.state import ensure_state, get_df


def _get_valid_cols(df: pd.DataFrame) -> list[str]:
    # All columns could potentially be labels, but usually int/str
    return list(df.columns)


def main() -> None:
    ensure_state()
    st.set_page_config(page_title="Evaluation - Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")
    ui_utils.load_app_style()

    st.title("评估：聚类指标计算")
    st.caption("选择 Ground Truth 列和预测标签列，计算 Pairwise F1, BCubed F1 等指标。")

    df = get_df()
    if df is None:
        st.info("请先在 Home 页面加载数据。")
        return

    cols = _get_valid_cols(df)
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            gt_col = st.selectbox(
                "Ground Truth 列 (GT)", 
                options=cols, 
                index=cols.index("gt_person_id") if "gt_person_id" in cols else 0
            )
        with c2:
            # Default to a cluster column if available
            default_pred = 0
            for i, c in enumerate(cols):
                if c.startswith("cluster_id"):
                    default_pred = i
                    break
            pred_col = st.selectbox("预测标签列 (Prediction)", options=cols, index=default_pred)

        run_btn = st.button("开始评估", type="primary", use_container_width=True)

    if run_btn:
        if gt_col == pred_col:
            st.error("GT 列和预测列不能相同，请选择不同的列。")
            return

        # Prepare data: drop rows where either label is NaN
        # (Alternatively, fillna with a special 'unassigned' label, but dropping is safer for strict eval)
        mask = df[gt_col].notna() & df[pred_col].notna()
        valid_df = df[mask]
        n_dropped = len(df) - len(valid_df)
        
        if len(valid_df) == 0:
            st.error("没有有效样本（GT 或 Pred 存在缺失值后为空）。")
            return

        gt_arr = valid_df[gt_col].values
        pred_arr = valid_df[pred_col].values

        st.divider()
        st.markdown(f"**评估样本数**: {len(valid_df):,} (已忽略 {n_dropped:,} 条含 NaN 的样本)")

        # 1. Pairwise
        t0 = time.time()
        pw_res = eval_utils.pairwise_f1(gt_arr, pred_arr)
        t_pw = time.time() - t0

        # 2. BCubed
        t0 = time.time()
        bc_res = eval_utils.bcubed_f1(gt_arr, pred_arr)
        t_bc = time.time() - t0
        
        # 3. Custom Eval
        t0 = time.time()
        custom_res = eval_utils.custom_eval(gt_arr, pred_arr)
        t_custom = time.time() - t0
        
        # 4. Basic Stats
        gt_stats = eval_utils.compute_basic_stats(gt_arr)
        pred_stats = eval_utils.compute_basic_stats(pred_arr)

        # Render Results
        
        st.subheader("核心指标 (F1 Score)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Pairwise F1", f"{pw_res['f1']:.4f}", help="基于成对关系的 F1")
        m2.metric("BCubed F1", f"{bc_res['f1']:.4f}", help="基于个体的 F1 (Recall/Precision 加权)")
        m3.metric("Custom F1", f"{custom_res['f1']:.4f}", help="自定义轨迹/封面照逻辑 F1")
        
        with st.expander("查看详细指标 (Precision/Recall)", expanded=True):
            res_df = pd.DataFrame([
                {"Metric": "Pairwise", "Precision": pw_res["precision"], "Recall": pw_res["recall"], "F1": pw_res["f1"], "Time(s)": t_pw},
                {"Metric": "BCubed", "Precision": bc_res["precision"], "Recall": bc_res["recall"], "F1": bc_res["f1"], "Time(s)": t_bc},
                {"Metric": "Custom", "Precision": custom_res["precision"], "Recall": custom_res["recall"], "F1": custom_res["f1"], "Time(s)": t_custom},
            ])
            st.dataframe(res_df.style.format("{:.4f}", subset=["Precision", "Recall", "F1", "Time(s)"]), use_container_width=True)

        st.subheader("簇统计对比")
        c_gt, c_pred = st.columns(2)
        with c_gt:
            st.markdown(f"**GT (`{gt_col}`)**")
            st.json(gt_stats)
        with c_pred:
            st.markdown(f"**Pred (`{pred_col}`)**")
            st.json(pred_stats)


if __name__ == "__main__":
    main()
