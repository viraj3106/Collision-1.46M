import os
import sys
from fpdf import FPDF

class ScalingReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "COLLISION-3M Scaling Experiment Results", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", border=0, align="C")

def main():
    pdf = ScalingReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(31, 41, 55) # Dark gray
    pdf.cell(0, 15, "COLLISION-3M Scaling Experiment Results", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(229, 231, 235) # Light gray separator line
    pdf.line(10, 25, 200, 25)
    pdf.ln(5)
    
    # Overview Description
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(75, 85, 99)
    desc_txt = (
        "This document presents the results of the COLLISION-3M scaling experiment, "
        "comparing it directly against the baseline COLLISION-1.46M model. Both models were "
        "trained on the identical dataset (collision_dataset_v4) under identical hyperparameter "
        "conditions on CPU for exactly 1,536,000 training tokens (1,500 steps)."
    )
    pdf.multi_cell(0, 5, desc_txt, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Direct Comparison Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 8, "Direct Comparison Metrics", new_x="LMARGIN", new_y="NEXT")
    
    headers = ["Metric", "COLLISION-1.46M (Base)", "COLLISION-3M (Exp)", "Change (%)"]
    rows = [
        ["Model Parameters", "1,462,464", "3,375,680", "+130.82%"],
        ["Best Val Loss", "1.9363", "0.9663", "-50.10%"],
        ["Best Val Perplexity", "6.93", "2.63", "-62.08%"],
        ["Total Train Time (s)", "490.5s", "1067.4s", "+117.62%"],
        ["Avg Inference (tok/s)", "200.83", "92.12", "-54.13%"]
    ]
    
    col_widths = [60, 45, 45, 40]
    
    # Header styling
    pdf.set_fill_color(243, 244, 246) # Light grey header fill
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "B", 10)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 10)
    for r in rows:
        # Highlight best val loss/perp
        highlight = "Best Val" in r[0]
        if highlight:
            pdf.set_fill_color(239, 246, 255) # Light blue box for key findings
            pdf.set_font("Helvetica", "B", 10)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 10)
            
        for i, val in enumerate(r):
            align = "L" if i == 0 else "C"
            pdf.cell(col_widths[i], 8, val, border=1, align=align, fill=True)
        pdf.ln()
        
    pdf.ln(5)
    
    # Observations Note Box
    pdf.set_fill_color(236, 253, 245) # Light green box for positive note
    pdf.set_font("Helvetica", "I", 9)
    note_txt = "Note: Parameter capacity scaling (+130.8%) yielded a significant 50.1% decrease in Validation Loss and a 62.1% decrease in Perplexity under identical training steps."
    pdf.cell(0, 8, note_txt, border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    
    # Generation Benchmark
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 8, "Generation Benchmark Outputs (COLLISION-3M)", new_x="LMARGIN", new_y="NEXT")
    
    gens = [
        ("What is artificial intelligence?", 
         "What is artificial intelligence? this and intelligence sequence and and maximize necessary intelligence next maximize loss. to and data domain.\n\nTransformers self-attention adapts hidden maintaining",
         "82.66 tok/s | 50 tokens"),
        ("The future of technology", 
         "The future of technology and and and and and comus methods.\nKey and partics partilasures the s and to toclaccape",
         "98.91 tok/s | 50 tokens")
    ]
    
    for prompt, text, stats in gens:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 6, f"Prompt: \"{prompt}\"  ({stats})", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_fill_color(249, 250, 251) # Off-white background
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(75, 85, 99)
        pdf.multi_cell(0, 5, text, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        
    pdf.ln(5)
    
    # Page Break for Curve
    pdf.add_page()
    
    # Loss Curve Image
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 8, "COLLISION-3M Loss Curve", new_x="LMARGIN", new_y="NEXT")
    
    image_path = "experiments/scaling/collision_3m/loss_curve.png"
    if os.path.exists(image_path):
        pdf.image(image_path, x=20, y=None, w=170)
        pdf.ln(10)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 8, "[Loss curve image not found]", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    # Save output
    output_pdf_path = "experiments/scaling/collision_3m/results.pdf"
    pdf.output(output_pdf_path)
    print(f"PDF saved successfully to {output_pdf_path}")

if __name__ == "__main__":
    main()
