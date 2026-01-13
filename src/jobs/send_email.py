import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from src.common.logging import setup_logger
from src.common.timeutil import today_et

log = setup_logger("email")


def attach_csv(msg: EmailMessage, path: Path):
  msg.add_attachment(path.read_bytes(), maintype="text", subtype="csv", filename=path.name)


def main():
  date = today_et()
  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / date
  action = out_dir / "action_list.csv"
  watch = out_dir / "watch_list.csv"
  if not action.exists() or not watch.exists():
    raise RuntimeError(f"Missing report files in {out_dir}")

  msg = EmailMessage()
  msg["Subject"] = f"[Action List] {date}"
  msg["From"] = os.environ["EMAIL_FROM"]
  msg["To"] = os.environ["EMAIL_TO"]
  msg.set_content(f"Daily report {date}. Attached action_list.csv and watch_list.csv.")

  attach_csv(msg, action)
  attach_csv(msg, watch)

  host = os.environ["SMTP_HOST"]
  port = int(os.environ.get("SMTP_PORT", "587"))
  user = os.environ["SMTP_USER"]
  pwd = os.environ["SMTP_PASS"]

  log.info("Sending email...")
  with smtplib.SMTP(host, port, timeout=30) as s:
    s.starttls()
    s.login(user, pwd)
    s.send_message(msg)
  log.info("Email sent.")


if __name__ == "__main__":
  main()
