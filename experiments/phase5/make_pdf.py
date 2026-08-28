import os
from fpdf import FPDF

class TrainingReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "Test Training 1: COLLISION-1.46M Phase 5 Run", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", border=0, align="C")

def main():
    pdf = TrainingReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(31, 41, 55) # Dark gray
    pdf.cell(0, 15, "Test Training 1: COLLISION-1.46M Phase 5", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(229, 231, 235) # Light gray separator line
    pdf.line(10, 25, 200, 25)
    pdf.ln(5)
    
    # Quick Stats
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Quick Statistics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(75, 85, 99)
    
    stats = [
        ("Dataset Version", "collision_dataset_v3"),
        ("Total Tokens", "2,411,502 (Train: 2,108,753, Val: 302,749)"),
        ("Model Parameters", "1,462,464 (3 layers, 128 d_model, 4 heads)"),
        ("Vocabulary Capacity", "8,000 capacity (890 active tokens)"),
        ("Steps Trained", "2,000 steps on CPU")
    ]
    
    for label, val in stats:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, f"{label}:", border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, val, border=0, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(8)
    
    # Table Header for metrics
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Learning Metrics", new_x="LMARGIN", new_y="NEXT")
    
    headers = ["Step", "Train Loss", "Val Loss", "Perplexity", "Learning Rate", "Time (s)"]
    row_data = [
        ["0 (Initial)", "-", "5.1501", "172.45", "0.000000", "0.0"],
        ["500", "4.4737", "5.1501", "172.45", "0.000375", "199.2"],
        ["1000", "2.8216", "4.3478", "77.31", "0.000599", "409.8"],
        ["1500", "2.0934", "4.1409", "62.86", "0.000588", "612.9"],
        ["2000 (Final)", "1.4061", "4.3394", "76.66", "0.000564", "817.8"]
    ]
    
    # Set table styling
    pdf.set_fill_color(243, 244, 246) # Light grey header fill
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "B", 10)
    
    col_widths = [30, 30, 30, 30, 40, 30]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align="C", fill=True)
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 10)
    for row in row_data:
        # Highlight best val loss step (1500)
        is_best = (row[0] == "1500")
        if is_best:
            pdf.set_fill_color(236, 253, 245) # Light green highlight
            pdf.set_font("Helvetica", "B", 10)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 10)
            
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 8, val, border=1, align="C", fill=True)
        pdf.ln()
        
    pdf.ln(5)
    
    # Validation Note Box
    pdf.set_fill_color(239, 246, 255) # Light blue box for info note
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 10, "Note: The best validation loss was 4.1409 achieved at step 1500 (perplexity: 62.86).", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Page Break for Generation and Loss Curve
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Generation Progression", new_x="LMARGIN", new_y="NEXT")
    
    gens = [
        ("Prompt: \"Artificial intelligence\"", 
         "Baseline (Step 0): \"Artificial intelligencelabSus \"\n"
         "Step 500: \"Artificial intelligence a syeto ti  i.isw er ,e a ofue esof . . .to  h hbo ymc.\"\n"
         "Step 1000: \"Artificial intelligence daily of Section this applications of Section ddele rimagnetifergy the ced of celject m ofs of\"\n"
         "Step 2000: \"Artificial intelligence claims. Document Section provide Section supporting methodologies. Section 7: Darra An of Experts field. Recent The Elation Ke b\"")
    ]
    
    for prompt, output in gens:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 6, prompt, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_fill_color(249, 250, 251) # Off-white background
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(75, 85, 99)
        # Use multi_cell for paragraph text
        pdf.multi_cell(0, 5, output, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    
    # Loss Curve Image
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Loss Curve", new_x="LMARGIN", new_y="NEXT")
    
    image_path = "C:/Users/viraj/.gemini/antigravity-ide/brain/c5b0d5f8-3e0b-46c4-89eb-c91e59977e2b/loss_curve.png"
    if os.path.exists(image_path):
        pdf.image(image_path, x=20, y=None, w=170)
        pdf.ln(10)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "[Loss curve image not found]", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    # Classification Box
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 10, "Training Classification", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_fill_color(209, 250, 229) # Vibrant light green for HEALTHY
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(6, 95, 70) # Dark green text
    pdf.cell(0, 8, "HEALTHY", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(75, 85, 99)
    desc = "Both training and validation loss decreased substantially over the 2,000 steps compared to baseline. A slight upward trend in validation loss between steps 1500 and 2000 suggests the onset of overfitting, meaning that 1,500 steps is the optimal training length for this model/dataset configuration."
    pdf.multi_cell(0, 5, desc, new_x="LMARGIN", new_y="NEXT")
    
    # Save output
    output_pdf_path = "C:/Users/viraj/.gemini/antigravity-ide/brain/c5b0d5f8-3e0b-46c4-89eb-c91e59977e2b/test_training_1.pdf"
    pdf.output(output_pdf_path)
    print(f"PDF saved successfully to {output_pdf_path}")

if __name__ == "__main__":
    main()
