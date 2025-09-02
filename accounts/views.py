# accounts/views.py
from django.contrib.auth import get_user_model
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .serializers import RegisterSerializer, UserSerializer
from .tokens import email_verification_token
from .utils import send_verification_email
from rest_framework_simplejwt.tokens import RefreshToken
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django_ratelimit.decorators import ratelimit


User = get_user_model()

class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    # Simple rate limit decorator to reduce abuse (5 reqs per minute per ip)
    @method_decorator(ratelimit(key="ip", rate="5/m", method='POST', block=True), name='dispatch')
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            # send verification email
            send_verification_email(request, user)
            return Response(
                {"message": "User registered. Please verify your email."},
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class VerifyEmailView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        uidb64 = request.GET.get("uid")
        token = request.GET.get("token")
        if not uidb64 or not token:
            return Response({"detail": "Invalid verification link."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None
        if user and email_verification_token.check_token(user, token):
            user.is_email_verified = True
            user.save()
            return Response({"message": "Email verified successfully."})
        return Response({"detail": "Invalid or expired token."}, status=status.HTTP_400_BAD_REQUEST)


class ResendVerificationView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        email = request.data.get("email")
        if not email:
            return Response({"email": "This field is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(email__iexact=email)
        except User.DoesNotExist:
            return Response({"detail": "User not found."}, status=status.HTTP_404_NOT_FOUND)
        if user.is_email_verified:
            return Response({"detail": "Email already verified."}, status=status.HTTP_400_BAD_REQUEST)
        send_verification_email(request, user)
        return Response({"message": "Verification email resent."})


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        identifier = request.data.get("identifier")  # username or email
        password = request.data.get("password")
        if not identifier or not password:
            return Response({"detail": "Missing credentials."}, status=status.HTTP_400_BAD_REQUEST)

        # Try username, then email
        try:
            user = User.objects.get(username__iexact=identifier)
        except User.DoesNotExist:
            try:
                user = User.objects.get(email__iexact=identifier)
            except User.DoesNotExist:
                user = None

        if user is None:
            return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        if not user.check_password(password):
            return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        # Optionally enforce email verification before login
        # if not user.is_email_verified:
        #     return Response({"detail": "Email not verified."}, status=status.HTTP_403_FORBIDDEN)

        refresh = RefreshToken.for_user(user)
        access = str(refresh.access_token)
        # Return shape expected by frontend: { token: <access>, refresh: <refresh>, user: {...} }
        return Response({
            "token": access,
            "refresh": str(refresh),
            "user": UserSerializer(user).data
        })


class MeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)
