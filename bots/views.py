# bots/views.py
from rest_framework import generics, permissions
from .models import Bot
# A linha de importação agora vai funcionar corretamente
from .serializers import BotSerializer, BotDetailSerializer 

class BotListCreateView(generics.ListCreateAPIView):
    """
    API view for listing and creating bots.
    """
    queryset = Bot.objects.all()
    serializer_class = BotSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        # Assign the current user as the owner of the bot.
        serializer.save(owner=self.request.user)

class BotDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    API view for retrieving, updating, and deleting a bot.
    Uses the detailed serializer for GET requests.
    """
    queryset = Bot.objects.all()
    serializer_class = BotDetailSerializer 
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Allow any authenticated user to view details, 
        # the 'createdByMe' field in the serializer will handle ownership logic.
        return Bot.objects.all()