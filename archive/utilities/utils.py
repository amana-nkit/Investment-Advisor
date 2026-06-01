from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import smtplib
from email.message import EmailMessage
import os

def create_pdf(text, filename="proposal.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750

    for line in text.split("\n"):
        c.drawString(40, y, line[:100])
        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()
    return filename

def send_email(to_email, file_path):
    msg = EmailMessage()
    msg["Subject"] = "Investment Proposal"
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to_email
    msg.set_content("Attached is your proposal")

    with open(file_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename="proposal.pdf"
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)