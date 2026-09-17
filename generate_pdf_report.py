import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (on pages after cover page)
        if self._pageNumber > 1:
            self.drawString(54, 750, "COLLISION Series — 1M to 10M Parameter Scaling Technical Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — COLLISION RESEARCH LAB")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        
        self.restoreState()

def build_pdf():
    pdf_filename = "COLLISION_1M_to_10M_Scaling_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=14,
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E3A8A"),
        backColor=colors.HexColor("#EFF6FF"),
        borderColor=colors.HexColor("#BFDBFE"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=12,
        borderRadius=4
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F8FAFC"),
        borderColor=colors.HexColor("#E2E8F0"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=0
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # Title & Header Block
    story.append(Paragraph("COLLISION SERIES: 1M ➔ 10M SCALING REPORT", title_style))
    story.append(Paragraph("A 70-Phase Causal Language Modeling Study in Extreme Low-Resource Regimes", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=12))

    # Metadata Table
    meta_data = [
        [Paragraph("<b>Project:</b> COLLISION", table_cell_style), Paragraph("<b>Target Scale:</b> 1.46M to 10.28M Params", table_cell_style)],
        [Paragraph("<b>Hardware Regime:</b> Consumer CPU (From Scratch)", table_cell_style), Paragraph("<b>Release Version:</b> COLLISION-10M v1.0.0", table_cell_style)],
        [Paragraph("<b>Hugging Face Hub:</b> collision-10M/collision-10m", table_cell_style), Paragraph("<b>Status:</b> DATA_COLLECTION_HOLD (Phase 70)", table_cell_style)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1"))
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "COLLISION is an open research investigation into Transformer causal language modeling operating under extreme resource constraints. "
        "The project demonstrates that high-quality data engineering, deterministic split protocols, and rigorous verification gates can produce stable, "
        "convergent representation spaces at sub-50M parameter scales on consumer CPUs without GPU pretraining.", body_style
    ))
    story.append(Paragraph(
        "<b>Core Research Hypothesis:</b> Data quality engineering, leakage elimination, and clean evaluation protocols drive model convergence more effectively than parameter expansion alone in low-resource regimes.", callout_style
    ))

    # Key Milestones Table
    story.append(Paragraph("2. Evolution & Benchmark Progress (Phases 1 – 70)", h1_style))
    
    headers = [Paragraph("Phase / Variant", table_header_style), Paragraph("Params", table_header_style), Paragraph("Val Perplexity", table_header_style), Paragraph("Key Technical Milestone", table_header_style)]
    data = [headers,
        [Paragraph("Phase 5 (Baseline)", table_cell_style), Paragraph("1.46M", table_cell_style), Paragraph("62.86", table_cell_style), Paragraph("Severe overfitting & alphabetical split bias", table_cell_style)],
        [Paragraph("Phase 6 (Data Audit)", table_cell_style), Paragraph("1.46M", table_cell_style), Paragraph("6.93", table_cell_style), Paragraph("Dataset v4: Subject-wise split (0% leakage) dropped perplexity by 55.9 pts", table_cell_style)],
        [Paragraph("Phase 10 (Scaling)", table_cell_style), Paragraph("3.38M", table_cell_style), Paragraph("3.75", table_cell_style), Paragraph("Capacity expansion validation on CPU", table_cell_style)],
        [Paragraph("Phase 13 (Instruct Test)", table_cell_style), Paragraph("3.38M", table_cell_style), Paragraph("Degraded", table_cell_style), Paragraph("Synthetic instruction crash (Repetition 57.6%)", table_cell_style)],
        [Paragraph("Phase 15 (Flagship Base)", table_cell_style), Paragraph("10.28M", table_cell_style), Paragraph("2.11", table_cell_style), Paragraph("COLLISION-10M pretraining on 10M tokens (Test Perplexity: 1.79)", table_cell_style)],
        [Paragraph("Phase 43 (DPO Audit)", table_cell_style), Paragraph("10.28M", table_cell_style), Paragraph("Loss: 3.19", table_cell_style), Paragraph("Discovered missing reference ratio in DPO math", table_cell_style)],
        [Paragraph("Phase 52 (SFT Candidate)", table_cell_style), Paragraph("10.28M", table_cell_style), Paragraph("Gen Score: 66.85", table_cell_style), Paragraph("Apex Candidate J52 promoted (Coherence: 38.50)", table_cell_style)],
        [Paragraph("Phase 70 (Final Release)", table_cell_style), Paragraph("10.28M", table_cell_style), Paragraph("0 Defects", table_cell_style), Paragraph("Full telemetry & DATA_COLLECTION_HOLD verdict", table_cell_style)]
    ]
    t = Table(data, colWidths=[100, 50, 84, 270])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Deep Dive Sections
    story.append(Paragraph("3. Data Quality vs. Parameter Capacity Breakthrough", h1_style))
    story.append(Paragraph(
        "During Phase 5 pretraining of COLLISION-1.46M, validation perplexity stalled at 62.86. A comprehensive forensic audit revealed two structural errors: "
        "1) Alphabetical document splitting caused complete domain missingness in training splits, and 2) 26.82% contiguous sentence leakage existed between train and val. "
        "In Phase 6, constructing <code>collision_dataset_v4</code> with subject-wise deterministic splits reduced validation perplexity to <b>6.93</b> without changing model architecture.", body_style
    ))

    story.append(Paragraph("4. The DPO Failure & Mathematical Bug Discovery", h1_style))
    story.append(Paragraph(
        "In Phase 42–46, Direct Preference Optimization (DPO) was explored. Initial DPO candidates suffered coherence collapse. "
        "Phase 43 forensic auditing discovered a critical code defect: the loss function omitted the reference model ratio pi_ref, penalizing policy log-probs without KL anchoring. "
        "Even after repairing canonical DPO, experiments proved that standard DPO preference rewards penalize 10M models due to length/repetition sensitivity.", body_style
    ))

    story.append(Paragraph("5. Supervised Fine-Tuning & Apex Candidate J52", h1_style))
    story.append(Paragraph(
        "Pivoting to Supervised Fine-Tuning (SFT) yielded the highest performance. Apex Candidate <b>J52</b> (Phase 52) trained on <code>collision_sft_v3</code> achieved: "
        "<b>Generalization Score: 66.85, Coherence: 38.50, Instruction Following: 48.20</b>.", body_style
    ))

    story.append(Paragraph("6. Public Beta, Telemetry & Hugging Face Release", h1_style))
    story.append(Paragraph(
        "Phases 53–70 introduced a production FastAPI server, Streamlit client (COLLISION LAB), PII redaction engine, and automated readiness gates requiring 100 clean real-world human records. "
        "Per strict scientific guidelines, zero synthetic user data was generated, placing the project into an explicit <code>DATA_COLLECTION_HOLD_HUMAN_TRAFFIC_REQUIRED</code> status. "
        "COLLISION-10M v1.0.0 is officially published on Hugging Face at <b>collision-10M/collision-10m</b> with verified SHA256 <code>d256d46d...3775b97</code>.", body_style
    ))

    # Code Block
    story.append(Paragraph("Quick Hugging Face / Python Inference Example:", h2_style))
    story.append(Paragraph(
        "python release_inference.py --prompt \"Artificial intelligence is\" --checkpoint models/collision-10m/model.pt", code_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
