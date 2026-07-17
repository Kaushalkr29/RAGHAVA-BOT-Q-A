from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

class Agent:

    def __init__(
        self,
        smtp_server,
        smtp_port
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def create_pdf(self, content, pdf_file="output.pdf"):
        """
        Create PDF from content string.
        """
        c = canvas.Canvas(pdf_file, pagesize=letter)

        width, height = letter
        y = height - 40

        for line in content.split("\n"):
            c.drawString(40, y, line)
            y -= 20

            if y < 40:
                c.showPage()
                y = height - 40

        c.save()

        print(f"PDF created: {pdf_file}")
        return pdf_file

    def create_text_file(self, content, txt_file="output.txt"):
        """
        Create TXT file from content string.
        """
        with open(txt_file, "w", encoding="utf-8") as file:
            file.write(content)

        print(f"Text file created: {txt_file}")
        return txt_file

    def send_email(self, receiver_email, content):
        """
        Send email using content as email body.
        """
        self.sender_email="kaushalk42k@gmail.com"
        load_dotenv()
        self.sender_password=os.getenv("password")
        msg = MIMEMultipart()

        msg["From"] = self.sender_email
        msg["To"] = receiver_email
        msg["Subject"] = "Generated Content"

        msg.attach(MIMEText(content, "plain"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(
                    self.sender_email,
                    self.sender_password
                )

                server.send_message(msg)

            print("Email sent successfully")

        except Exception as e:
            print(f"Email sending failed: {e}")