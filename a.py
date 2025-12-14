from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import mplcursors


def preprocess_traffic_data(site_loc_path: str, traffic_data_path: str):

    site_loc = pd.read_csv(site_loc_path)
    traffic_data = pd.read_csv(traffic_data_path)

    traffic_data["Date"] = pd.to_datetime(traffic_data[["Year", "Month", "Day"]])
    traffic_data = traffic_data.dropna(subset=["Site"])

    traffic_data["SiteID"] = traffic_data["Site"].astype(str)
    site_loc["SiteID"] = site_loc["SiteID"].astype(str)

    site_loc = site_loc.rename(columns={"Lat": "Latitude", "Long": "Longitude"})
    site_loc["Longitude"] = pd.to_numeric(site_loc["Longitude"], errors="coerce")
    site_loc["Latitude"] = pd.to_numeric(site_loc["Latitude"], errors="coerce")
    site_loc = site_loc.dropna(subset=["Longitude", "Latitude"])

    # Keep only plausible Dublin-area coordinates to avoid outliers like (0, 0)
    site_loc = site_loc[
        site_loc["Longitude"].between(-7, -5) & site_loc["Latitude"].between(52, 54)
    ]

    site_daily_flow = traffic_data[["SiteID", "Date", "SumOfSumVolume"]].rename(
        columns={"SumOfSumVolume": "total_flow"}
    )

    site_avg_flow = (
        site_daily_flow.groupby("SiteID", as_index=False)["total_flow"]
        .mean()
        .rename(columns={"total_flow": "avg_daily_flow"})
    )

    site_geo_flow = pd.merge(site_loc, site_avg_flow, on="SiteID", how="inner")

    flow_pivot = site_daily_flow.pivot(
        index="Date", columns="SiteID", values="total_flow"
    ).fillna(0)
    corr_matrix = flow_pivot.corr()

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
    """Plot (1) site distribution on an OSM basemap and (2) Top-20 correlation heatmap."""
    fig = plt.figure(figsize=(18, 8))

    tiler = cimgt.OSM()
    mercator = tiler.crs
    ax1 = fig.add_subplot(121, projection=mercator)

    dublin_extent = [-6.35, -6.2, 53.32, 53.42]
    ax1.set_extent(dublin_extent, crs=ccrs.PlateCarree())
    ax1.add_image(tiler, 13)

    scatter = ax1.scatter(
        site_geo_flow["Longitude"],
        site_geo_flow["Latitude"],
        c=site_geo_flow["avg_daily_flow"],
        cmap="Reds",
        s=np.clip(site_geo_flow["avg_daily_flow"] / 50, 10, 500),
        alpha=0.7,
        edgecolors="black",
        linewidth=0.4,
        transform=ccrs.PlateCarree(),
    )

    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.02)
    cbar1.set_label("Average daily flow", fontsize=10)
    ax1.set_title("Dublin SCATS sites: spatial distribution (2020 Jan–Jun)", fontsize=12)

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        idx = sel.index
        site_id = str(site_geo_flow.iloc[idx]["SiteID"])
        avg_flow = int(site_geo_flow.iloc[idx]["avg_daily_flow"])
        sel.annotation.set_text(f"SiteID: {site_id}\nAvg daily flow: {avg_flow}")
        sel.annotation.get_bbox_patch().set_alpha(0.85)

    ax2 = fig.add_subplot(122)
    corr_sub = corr_matrix.loc[top_20_sites, top_20_sites]

    im = ax2.imshow(corr_sub.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax2.set_xticks(np.arange(len(top_20_sites)))
    ax2.set_yticks(np.arange(len(top_20_sites)))
    ax2.set_xticklabels(top_20_sites, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(top_20_sites, fontsize=8)
    ax2.set_title("Top-20 sites: traffic correlation heatmap", fontsize=12)

    cbar2 = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label("Pearson correlation", fontsize=10)

    for i in range(len(top_20_sites)):
        for j in range(len(top_20_sites)):
            ax2.text(
                j,
                i,
                f"{corr_sub.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    fig.suptitle("SCATS traffic: spatial distribution & correlation analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    SITE_LOC_PATH = "its_scats_sites_aug-2020.csv"
    TRAFFIC_DATA_PATH = "scats_detector_volume_jan-jun-2020.csv"

    try:
        print("Preprocessing data...")
        site_geo_flow_df, corr_matrix_df, top20_sites_list = preprocess_traffic_data(
            SITE_LOC_PATH, TRAFFIC_DATA_PATH
        )
        print("Preprocessing complete.")

        print("Generating plots...")
        visualize_traffic_geo_corr(site_geo_flow_df, corr_matrix_df, top20_sites_list)
        print("Done.")
    except FileNotFoundError as e:
        print(f"Error: data file not found - {e}")
    except Exception as e:
        print(f"Runtime error: {e}")

