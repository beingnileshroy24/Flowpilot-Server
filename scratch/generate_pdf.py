import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        # We don't want page numbers on the cover page
        if self._pageNumber == 1:
            return
            
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header
        self.drawString(54, 750, "Flowpilot Architecture & Design Specification")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 45, page_text)
        self.drawString(54, 45, "CONFIDENTIAL - INTERNAL DEVELOPMENT USE ONLY")
        self.line(54, 58, 558, 58)
        self.restoreState()

def build_pdf(filename="design.pdf"):
    # Target page layout margins: 0.75 in (54 pt)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=32,
        leading=38,
        textColor=colors.HexColor("#0f172a"),
        alignment=0, # Left aligned
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=16,
        leading=22,
        textColor=colors.HexColor("#475569"),
        spaceAfter=40
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b")
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"), # Dark blue
        spaceBefore=22,
        spaceAfter=12,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=10
    )

    story = []

    # ==================== PAGE 1: COVER PAGE ====================
    story.append(Spacer(1, 100))
    story.append(Paragraph("FLOWPILOT ENGINE", subtitle_style))
    story.append(Paragraph("Project Design & Architectural Specification", title_style))
    
    # Colored horizontal band accent
    band_data = [[""]]
    band_table = Table(band_data, colWidths=[504], rowHeights=[4])
    band_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#3b82f6")), # Blue accent
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(band_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("An AI-powered multi-tenant software project planning and task orchestration platform.", subtitle_style))
    story.append(Spacer(1, 150))
    
    meta_info = """
    <b>Document Version:</b> 1.1.0 (Task Board & Deletion Sync Updates)<br/>
    <b>Date:</b> July 16, 2026<br/>
    <b>Author:</b> Flowpilot Core Team & AI Coding Subagent<br/>
    <b>Target Environment:</b> macOS / Docker Containerized Dev Environment
    """
    story.append(Paragraph(meta_info, meta_style))
    story.append(PageBreak())

    # ==================== PAGE 2: ARCHITECTURE & DB ====================
    story.append(Paragraph("1. System Architecture Overview", h1_style))
    story.append(Paragraph("Flowpilot is organized as a decoupled modern web application consisting of a single-page React frontend running Vite, a FastAPI REST server, and a dual-database model integrating MongoDB for transactional state and LanceDB for vector workspace knowledge.", body_style))
    
    story.append(Paragraph("The transactional MongoDB database maps standard objects (users, projects, milestones, releases, tasks, and retro entries) which are mirrored asynchronously into a LanceDB vector database. When tasks are updated, deleted, or generated, changes trigger back-end events that align the indices automatically.", body_style))
    
    story.append(Paragraph("1.1. Data Models and Schemas", h2_style))
    story.append(Paragraph("Database schemas are built with Pydantic and beanie. Document relationships are scoped inside the <b>Project</b> context:", body_style))
    
    story.append(Paragraph("• <b>User Document:</b> Manages credentials and roles (ADMIN, MANAGER, LEAD_DEVELOPER, DEVELOPER, CLIENT). Scoped using standard bcrypt password hashing.", bullet_style))
    story.append(Paragraph("• <b>Project Document:</b> The core tenant bucket containing lists of embedded sub-models: Sprints (goals, capacity, duration), Milestones (targets), Releases (deployments), Decisions (Architecture Decision Records), and Retro Entries (went well, improvements, actions).", bullet_style))
    story.append(Paragraph("• <b>Task Document:</b> The ticket entity containing title, priority (LOW, MEDIUM, HIGH, CRITICAL), type (EPIC, TASK, SUBTASK, BUG), assignee_id, estimated_hours, sprint_id, and subtask checklist items.", bullet_style))
    story.append(Paragraph("• <b>Comment Document:</b> Threaded logs referenced to tasks.", bullet_style))
    
    story.append(Paragraph("1.2. Role-Based Access Control (RBAC)", h2_style))
    story.append(Paragraph("The platform enforces strict role validation at the router layer to prevent privilege escalations:", body_style))
    
    rbac_data = [
        ['Role', 'Project Config', 'Create Task', 'Update Task Status', 'Delete Task/Bulk'],
        ['Client', 'Read Only', 'Allowed', 'Blocked', 'Blocked'],
        ['Developer', 'Read Only', 'Allowed (Self Assign)', 'Allowed', 'Blocked'],
        ['Lead Dev', 'Allowed', 'Allowed (Assign Any)', 'Allowed', 'Blocked'],
        ['Manager', 'Allowed', 'Allowed (Assign Any)', 'Allowed', 'Allowed'],
        ['Admin', 'Allowed', 'Allowed (Assign Any)', 'Allowed', 'Allowed']
    ]
    rbac_table = Table(rbac_data, colWidths=[90, 100, 110, 100, 104])
    rbac_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(rbac_table)
    story.append(PageBreak())

    # ==================== PAGE 3: TASK BOARD & BULK OPERATIONS ====================
    story.append(Paragraph("2. Kanban Task Board & Bulk Operations", h1_style))
    story.append(Paragraph("The Kanban task board is the primary interface for managing project workflows. It divides issues into columns mapping directly to task states: <b>TODO</b>, <b>IN_PROGRESS</b>, <b>IN_REVIEW</b>, and <b>DONE</b>.", body_style))
    
    story.append(Paragraph("To support rapid development cycles, the platform implements unified bulk transitions and deletions, avoiding separate HTTP request overhead for each task card manipulation.", body_style))
    
    story.append(Paragraph("2.1. Backend API Implementations", h2_style))
    story.append(Paragraph("The FastAPI backend supports transactional bulk status updates and bulk task deletions via the following REST schemas:", body_style))
    
    code_bulk_status = """# Bulk status update endpoint
@router.post("/bulk-status")
async def bulk_update_task_status(payload: BulkStatusUpdateSchema, ...):
    # Triggers status mutations, updates updated_at, registers ActivityLogs,
    # and dispatches sync events to LanceDB asynchronously in the background.

# Bulk delete endpoint (Restricted to Manager/Admin)
@router.post("/bulk-delete")
async def bulk_delete_tasks(payload: BulkDeleteSchema, ...):
    # Performs transactional deletions, triggers background cleanup tasks,
    # and emits sync event 'delete' to mirror vector space changes."""
    story.append(Paragraph(code_bulk_status.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style))
    
    story.append(Paragraph("2.2. Frontend UI and Multi-Selection Flow", h2_style))
    story.append(Paragraph("The React Kanban board features a dual-layer select-all mechanism with custom overlays:", body_style))
    story.append(Paragraph("• <b>Board-Level Select All:</b> A checkbox situated in the main filter bar. Toggling it checks/unchecks all tasks currently matching the search filters.", bullet_style))
    story.append(Paragraph("• <b>Column-Level Select All:</b> Checkboxes in the header of each status column (e.g. Backlog, In Progress). Toggling check/uncheck selects only tasks inside that specific status column.", bullet_style))
    story.append(Paragraph("• <b>Floating Action Panel:</b> A glassmorphic sticky panel slide-in appears at the bottom center of the viewport once one or more checkboxes are checked. It exposes batch actions: To Do, In Prog, Review, Done, Delete, and Cancel. Deletions trigger a custom custom modal window confirming the actions in detail rather than browser defaults.", bullet_style))
    story.append(PageBreak())

    # ==================== PAGE 4: NEURAL WORK BREAKDOWN ENGINE & RAG ====================
    story.append(Paragraph("3. AI Copilot RAG & WBS Stream Engine", h1_style))
    story.append(Paragraph("The Flowpilot application integrates a closed-loop Retrieval-Augmented Generation (RAG) system with a deterministic routing loop to query system context, documents, and task boards directly.", body_style))
    
    story.append(Paragraph("3.1. Unified Knowledge Index and Retriever", h2_style))
    story.append(Paragraph("The platform operates a <b>HybridRetriever</b> blending keyword and vector search over a unified knowledge table inside LanceDB. When query context is loaded, elements undergo the following pipeline:", body_style))
    
    story.append(Paragraph("1. <b>Vector Search:</b> Embeds the user query via ModernBERT (768 dimensions) and queries LanceDB matching to an inclusive cosine similarity threshold (default 0.45). For backlog/task count searches, a lower threshold of 0.35 is applied to capture general intent.", bullet_style))
    story.append(Paragraph("2. <b>Keyword Search:</b> Tokenizes the query text and scans the local LanceDB index for overlap score, filtered strictly by target project_id and entity_type.", bullet_style))
    story.append(Paragraph("3. <b>Reciprocal Rank Fusion (RRF):</b> Merges rank indexes of both results to surface the most relevant elements.", bullet_style))
    
    story.append(Paragraph("3.2. Autorun MongoDB Authoritative Fallback", h2_style))
    story.append(Paragraph("Because semantic search can fail on numerical count questions (e.g. <i>'how many tasks are left to do?'</i>), the backlog search tool is designed with a MongoDB authoritative fetch. The engine always queries MongoDB for the authorative status lists, then blends it with vector search rankings. This guarantees 100% accurate status counts even if vector database indices are stale or rebuilding.", body_style))
    
    story.append(Paragraph("3.3. Work Breakdown Structure (WBS) Generation", h2_style))
    story.append(Paragraph("The WBS engine reads unstructured text requirements, feeds the context to the local LLM running on Apple Silicon, and outputs structural project task hierarchies in a JSON format streamed via Server-Sent Events (SSE). The generated WBS can be committed directly to create the project's task board backlog in one click.", body_style))

    # Render PDF
    doc.build(story, canvasmaker=NumberedCanvas)

if __name__ == "__main__":
    build_pdf()
