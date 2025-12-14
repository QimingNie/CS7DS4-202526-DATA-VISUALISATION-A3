# -*- coding: utf-8 -*-
"""
SCATS 站点车流空间分布 + 相关性分析（带真实地图底图）

依赖：
    pip install pandas numpy matplotlib cartopy mplcursors

数据文件（放在脚本同目录）：
    - its_scats_sites_aug-2020.csv
    - scats_detector_volume_jan-jun-2020.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import mplcursors

# 如果中文变成方块，可以改成 ['Microsoft YaHei'] 或其他本机存在的字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def preprocess_traffic_data(site_loc_path: str, traffic_data_path: str):
    """
    交通数据预处理函数（针对本次 SCATS 数据的精简版）：
    - 读取站点位置和日级流量数据
    - 构造日期字段
    - 计算各站点日均流量
    - 构建站点间相关性矩阵
    """
    # ---------- 读取数据 ----------
    site_loc = pd.read_csv(site_loc_path)
    traffic_data = pd.read_csv(traffic_data_path)

    # ---------- 构造 Date 字段（原始数据已是按日汇总） ----------
    traffic_data["Date"] = pd.to_datetime(
        traffic_data[["Year", "Month", "Day"]]
    )

    # 删除缺失站点 ID 的记录
    traffic_data = traffic_data.dropna(subset=["Site"])

    # 统一使用字符串类型的 SiteID，方便后续 join 和相关性矩阵索引
    traffic_data["SiteID"] = traffic_data["Site"].astype(str)
    site_loc["SiteID"] = site_loc["SiteID"].astype(str)

    # ---------- 清洗经纬度 ----------
    # 将 Lat / Long 重命名为 Latitude / Longitude
    site_loc = site_loc.rename(columns={"Lat": "Latitude", "Long": "Longitude"})
    site_loc["Longitude"] = pd.to_numeric(site_loc["Longitude"], errors="coerce")
    site_loc["Latitude"] = pd.to_numeric(site_loc["Latitude"], errors="coerce")
    site_loc = site_loc.dropna(subset=["Longitude", "Latitude"])

    # 粗略过滤出都柏林附近的站点，避免 (0,0) 等异常坐标
    site_loc = site_loc[
        (site_loc["Longitude"].between(-7, -5))
        & (site_loc["Latitude"].between(52, 54))
    ]

    # ---------- 构造日级流量表 ----------
    # 本数据已经是“每天一行”，直接使用 SumOfSumVolume 作为 daily flow
    site_daily_flow = traffic_data[["SiteID", "Date", "SumOfSumVolume"]].rename(
        columns={"SumOfSumVolume": "total_flow"}
    )

    # ---------- 计算每站点日均流量 ----------
    site_avg_flow = (
        site_daily_flow.groupby("SiteID", as_index=False)["total_flow"]
        .mean()
        .rename(columns={"total_flow": "avg_daily_flow"})
    )

    # ---------- 合并空间位置与日均流量 ----------
    site_geo_flow = pd.merge(site_loc, site_avg_flow, on="SiteID", how="inner")

    # ---------- 构建站点车流相关性矩阵 ----------
    # 透视为 Date × SiteID 的矩阵，对列之间计算皮尔逊相关系数
    flow_pivot = site_daily_flow.pivot(
        index="Date", columns="SiteID", values="total_flow"
    ).fillna(0)
    corr_matrix = flow_pivot.corr()

    # ---------- 选取流量前 20 的站点 ----------
    top_20_sites = (
        site_avg_flow.sort_values("avg_daily_flow", ascending=False)
        .head(20)["SiteID"]
        .tolist()
    )

    return site_geo_flow, corr_matrix, top_20_sites


def visualize_traffic_geo_corr(
    site_geo_flow: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    top_20_sites: list[str],
):
    """
    绘制地理空间分布 + TOP20 站点相关性热力图（带 OSM 底图）
    """
    fig = plt.figure(figsize=(18, 8))

    # ---------- 左：真实地图 + 站点散点 ----------
    tiler = cimgt.OSM()           # OpenStreetMap 瓦片
    mercator = tiler.crs          # WebMercator 坐标系

    ax1 = fig.add_subplot(121, projection=mercator)

    # 都柏林范围（经纬度），注意 crs=PlateCarree
    dublin_extent = [-6.35, -6.2, 53.32, 53.42]
    ax1.set_extent(dublin_extent, crs=ccrs.PlateCarree())

    # 加载底图瓦片，zoom 值可调（11–14）
    ax1.add_image(tiler, 13)

    # 站点散点
    scatter = ax1.scatter(
        site_geo_flow["Longitude"],
        site_geo_flow["Latitude"],
        c=site_geo_flow["avg_daily_flow"],
        cmap="Reds",
        s=site_geo_flow["avg_daily_flow"] / 50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.4,
        transform=ccrs.PlateCarree(),
    )

    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.02)
    cbar1.set_label("日均车流量", fontsize=10)
    ax1.set_title("都柏林SCATS站点车流空间分布（2020.1–6）", fontsize=12, pad=10)

    # 悬停提示
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        idx = sel.index
        site_id = str(site_geo_flow.iloc[idx]["SiteID"])
        avg_flow = int(site_geo_flow.iloc[idx]["avg_daily_flow"])
        sel.annotation.set_text(f"站点ID: {site_id}\n日均流量: {avg_flow}")
        sel.annotation.get_bbox_patch().set_alpha(0.8)

    # ---------- 右：TOP20 站点相关性热力图 ----------
    ax2 = fig.add_subplot(122)

    corr_sub = corr_matrix.loc[top_20_sites, top_20_sites]

    im = ax2.imshow(corr_sub.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax2.set_xticks(np.arange(len(top_20_sites)))
    ax2.set_yticks(np.arange(len(top_20_sites)))
    ax2.set_xticklabels(top_20_sites, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(top_20_sites, fontsize=8)
    ax2.set_title("TOP20站点车流相关性热力图", fontsize=12, pad=10)

    cbar2 = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label("皮尔逊相关系数", fontsize=10)

    # 在格子中标出数值
    for i in range(len(top_20_sites)):
        for j in range(len(top_20_sites)):
            ax2.text(
                j,
                i,
                f"{corr_sub.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    fig.suptitle("SCATS站点车流空间分布与关联性分析", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # 根据实际路径修改
    SITE_LOC_PATH = "its_scats_sites_aug-2020.csv"
    TRAFFIC_DATA_PATH = "scats_detector_volume_jan-jun-2020.csv"

    try:
        print("开始预处理数据...")
        site_geo_flow_df, corr_matrix_df, top20_sites_list = preprocess_traffic_data(
            SITE_LOC_PATH, TRAFFIC_DATA_PATH
        )
        print("数据预处理完成！")

        print("开始生成可视化图表...")
        visualize_traffic_geo_corr(site_geo_flow_df, corr_matrix_df, top20_sites_list)
        print("图表生成完成！")
    except FileNotFoundError as e:
        print(f"错误：未找到指定数据文件 - {e}")
    except Exception as e:
        print(f"程序运行异常：{e}")
