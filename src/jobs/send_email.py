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
  date = os.environ.get("REPORT_DATE") or os.environ.get("EMAIL_DATE") or today_et()
  out_dir = Path(os.environ.get("OUTPUT_DIR", "/app/outputs")) / date
  summary = out_dir / "summary.txt"
  signal_csvs = sorted(out_dir.glob("signals_*.csv"))

  # Require at least one signals CSV (summary is optional but recommended)
  if not signal_csvs:
    raise RuntimeError(f"Missing signals_*.csv in {out_dir}")

  msg = EmailMessage()
  msg["Subject"] = f"[Signals] {date}"
  msg["From"] = os.environ["EMAIL_FROM"]
  msg["To"] = os.environ["EMAIL_TO"]

  # Email body: prefer summary.txt, otherwise a simple fallback
  if summary.exists():
    msg.set_content(summary.read_text(encoding="utf-8"))
  else:
    msg.set_content(f"Signals report {date}. Attached signals CSVs.")

  # Attach summary (as text) if present
  if summary.exists():
    msg.add_attachment(
      summary.read_bytes(),
      maintype="text",
      subtype="plain",
      filename="summary.txt",
    )

  # Attach all signals CSVs
  for p in signal_csvs:
    attach_csv(msg, p)

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
