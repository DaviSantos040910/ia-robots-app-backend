# bots/views.py
from rest_framework import generics, permissions
from .models import Bot
# Import both serializers
from .serializers import BotSerializer, BotDetailSerializer 

class BotListCreateView(generics.ListCreateAPIView):
    # ... (no changes here)
    queryset = Bot.objects.all()
    serializer_class = BotSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

class BotDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    API view for retrieving, updating, and deleting a bot.
    Uses the detailed serializer for GET requests.
    """
    queryset = Bot.objects.all()
    # Use the detailed serializer for retrieving
    serializer_class = BotDetailSerializer 
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Bot.objects.all() # Allow viewing any bot, permissions are handled by `createdByMe`