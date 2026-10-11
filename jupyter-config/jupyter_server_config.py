from pathlib import Path


logo_path = (
    Path(__file__).resolve().parents[1]
    / "extensions"
    / "error-assistant"
    / "style"
    / "modelmind-logo.png"
)

c.Branding.logo_uri = str(logo_path)
c.Branding.splash_logo_uri = str(logo_path)
c.Branding.short_name = "ModelMind Notebook"
