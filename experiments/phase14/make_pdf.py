import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# Adjust sys path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_CSV = "experiments/phase14/training_log.csv"
LOSS_CURVE = "experiments/phase14/loss_curve.png"
PDF_PATH = "experiments/phase14/results.pdf"

class ScalingReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "COLLISION Phase 14 Model Capacity Scaling Report", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", border=0, align="C")

def generate_plot():
    if not os.path.exists(LOG_CSV):
        print(f"Log CSV not found at {LOG_CSV}. Skipping plot.")
        return False
        
    df = pd.read_csv(LOG_CSV)
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["train_loss"], label="Train Loss", color="#ef4444", linewidth=2)
    plt.plot(df["step"], df["validation_loss"], label="Validation Loss", color="#3b82f6", linewidth=2)
    plt.title("COLLISION-10M Pretraining Loss Curve", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Training Step", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    # Stylize
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(LOSS_CURVE, dpi=300)
    plt.close()
    return True

def main():
    print("Generating loss curve plot...")
    generate_plot()
    
    print("Compiling PDF report...")
    pdf = ScalingReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(31, 41, 55) # Dark gray
    pdf.cell(0, 15, "COLLISION Phase 14: Model Capacity Scaling", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(229, 231, 235)
    pdf.line(10, 25, 200, 25)
    pdf.ln(5)
    
    # Quick Stats
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Quick Statistics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(75, 85, 99)
    
    stats = [
        ("Dataset Version", "collision_dataset_v5_expanded (Train: 1,546,977 tokens)"),
        ("Model Parameters", "10,282,304 (6 layers, 384 d_model, 8 heads, 768 d_ff)"),
        ("Vocabulary Capacity", "8,000 capacity (weight-tied embeddings)"),
        ("Steps Trained", "1,500 steps (from Random Initialization)"),
        ("Hardware / Device", "CPU-compatible training run")
    ]
    
    for label, val in stats:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, f"{label}:", border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, val, border=0, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(5)
    
    # Learning Metrics Table
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Learning Metrics Progression (10M Model)", new_x="LMARGIN", new_y="NEXT")
    
    headers = ["Step", "Train Loss", "Val Loss", "Perplexity", "Learning Rate", "Speed (tok/s)"]
    row_data = [
        ["500", "2.7877", "2.6326", "13.91", "0.000500", "803.5"],
        ["1000", "1.0413", "0.9119", "2.49", "0.000593", "876.6"],
        ["1500 (Final)", "0.3987", "0.4824", "1.62", "0.000564", "837.4"]
    ]
    
    pdf.set_fill_color(243, 244, 246)
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "B", 10)
    
    col_widths = [30, 30, 30, 30, 40, 30]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 10)
    for row in row_data:
        is_best = ("1500" in row[0])
        if is_best:
            pdf.set_fill_color(236, 253, 245) # Highlight green for final best
            pdf.set_font("Helvetica", "B", 10)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 10)
            
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 8, val, border=1, align="C", fill=True)
        pdf.ln()
        
    pdf.ln(5)
    
    # 3.38M vs 10M Comparative Table
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Model Capacity Comparison (3.38M vs 10M)", new_x="LMARGIN", new_y="NEXT")
    
    comp_headers = ["Metric", "3.38M Base Model", "10M Scaled Model"]
    comp_widths = [70, 60, 60]
    comp_rows = [
        ["Test Loss (Lower is Better)", "1.3213", "0.7679"],
        ["Test Perplexity (Lower is Better)", "3.75", "2.16"],
        ["Average Repetition Rate (Lower is Better)", "47.9%", "26.8%"],
        ["Average Unique Token Ratio (Higher is Better)", "52.1%", "73.2%"],
        ["Sentence Termination Rate (Higher is Better)", "55.6%", "88.9%"],
        ["Average Response Length", "76.1 tokens", "58.0 tokens"]
    ]
    
    pdf.set_fill_color(243, 244, 246)
    pdf.set_font("Helvetica", "B", 10)
    for i, h in enumerate(comp_headers):
        pdf.cell(comp_widths[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 10)
    for row in comp_rows:
        pdf.set_fill_color(255, 255, 255)
        for i, val in enumerate(row):
            align = "L" if i == 0 else "C"
            pdf.cell(comp_widths[i], 8, val, border=1, align=align, fill=True)
        pdf.ln()
        
    # Page Break for Generation Samples and Plotted curves
    pdf.add_page()
    
    # Generation progression
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Generation Coherence Samples", new_x="LMARGIN", new_y="NEXT")
    
    gens = [
        ("Prompt: \"Artificial intelligence is\"", 
         "3.38M BASE:\n"
         "  to digned to sw crestion: to configain dual)\n"
         "  Answer: maller to s to locaus by and dectromagneticement...\n\n"
         "10M BASE:\n"
         "  devel to maximize heatinforcement reward policies through reinforcement learning\n"
         "  policies where agents optimize behavior paths by maximizing rewards."),
        ("Prompt: \"To prevent overfitting, a model should\"",
         "3.38M BASE:\n"
         "  igit: cas transform therve where of of curvital distributions is stand...\n\n"
         "10M BASE:\n"
         "  ecision trees of to maximize information gain minimize Gini impurity values.")
    ]
    
    for prompt, output in gens:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 6, prompt, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_fill_color(249, 250, 251)
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(75, 85, 99)
        pdf.multi_cell(0, 5, output, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        
    pdf.ln(4)
    
    # Loss Curve Image
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Learning Loss Curve", new_x="LMARGIN", new_y="NEXT")
    
    if os.path.exists(LOSS_CURVE):
        pdf.image(LOSS_CURVE, x=15, w=180)
        pdf.ln(5)
        
    # Output to File
    pdf.output(PDF_PATH)
    print(f"Results PDF generated successfully at {PDF_PATH}!")

if __name__ == "__main__":
    main()
