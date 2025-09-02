# accounts/urls.py
from django.urls import path
from .views import RegisterView, VerifyEmailView, ResendVerificationView, LoginView, MeView
# views.py
from django_ratelimit.decorators import ratelimit  # ao invés de 'ratelimit.decorators'


urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("verify-email/", VerifyEmailView.as_view(), name="verify-email"),
    path("resend-verification/", ResendVerificationView.as_view(), name="resend-verification"),
    path("login/", LoginView.as_view(), name="login"),
    path("me/", MeView.as_view(), name="me"),
]
