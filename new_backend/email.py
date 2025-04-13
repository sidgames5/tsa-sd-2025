import smtplib
from email.mime.text import MIMEText

smtp_server = "smtp.gmail.com"
smtp_port = 465
smtp_email = "sid2.srinivasan@gmail.com"
smtp_password = "jfew mhdc orwi cgds"

def send_email(target,content="Scan Results from LeafLogic"):
    message = MIMEText(content, "html")
    message["Subject"] = "LeafLogic - Scan Complete"
    message["From"] = smtp_email
    message["To"] = target
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, target, message.as_string())
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
    
if __name__ == "__main__":
    send_email("Rubisco412@gmail.com", content="""
<html>

<body>
    <h1>LeafLogic</h1>
    <p>Your scan is ready</p>
</body>

</html>
               """)