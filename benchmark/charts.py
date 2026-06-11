import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import List
from benchmark.models import BenchmarkResult

def generate_charts(results: List[BenchmarkResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("No results to chart.")
        return
        
    df = pd.DataFrame([r.model_dump() for r in results])
    
    # 1. Processing Time Comparison
    fig1 = px.bar(df, x="config_id", y="processing_time_sec", color="config_id",
                  title="Processing Time by Configuration (Lower is Better)")
    fig1.update_layout(template="plotly_dark")
    fig1.write_html(str(output_dir / "processing_time.html"))
    
    # 2. FPS Comparison
    fig2 = px.bar(df, x="config_id", y="fps", color="config_id",
                  title="FPS by Configuration (Higher is Better)")
    fig2.update_layout(template="plotly_dark")
    fig2.write_html(str(output_dir / "fps_comparison.html"))
    
    # 3. RAM vs VRAM Usage
    fig3 = px.bar(df, x="config_id", y=["peak_ram_mb", "peak_vram_mb"], barmode="group",
                  title="Memory Usage (MB)")
    fig3.update_layout(template="plotly_dark")
    fig3.write_html(str(output_dir / "memory_usage.html"))
    
    print(f"Charts generated in {output_dir}")
