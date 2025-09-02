# accounts/utils.py
from django.conf import settings
from django.urls import reverse
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from .tokens import email_verification_token

def send_verification_email(request, user):
    """
    Send an email with a verification link.
    In dev we print to console; in production configure SMTP.
    """
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = email_verification_token.make_token(user)
    # Build URL (you may use a frontend deep link or a web page that calls API verify)
    # Example: https://your-domain.com/verify-email/?uid=...&token=...
    verify_path = f"/api/auth/verify-email/?uid={uid}&token={token}"
    # If you have a frontend URL for opening, prefix accordingly
    verify_url = f"{request.scheme}://{request.get_host()}{verify_path}"
    subject = "Verify your email"
    message = f"Hello {user.username},\n\nPlease verify your email by clicking the link below:\n{verify_url}\n\nIf you didn't sign up, ignore this email."
    send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [user.email])
