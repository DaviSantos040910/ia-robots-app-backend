from django.shortcuts import render

# Create your views here.
# explore/views.py
from rest_framework import generics, status
from rest_framework.response import Response
from bots.models import Bot, Category
from bots.serializers import BotSerializer, CategorySerializer
from .models import SearchHistory
from .serializers import SearchHistorySerializer
from rest_framework.permissions import IsAuthenticated

class ExploreCategoryListView(generics.ListAPIView):
    """View to list all categories for the explore screen."""
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [IsAuthenticated]

class ExploreBotListView(generics.ListAPIView):
    """View to list public bots, optionally filtered by category or search term."""
    serializer_class = BotSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset = Bot.objects.filter(publicity=Bot.Publicity.PUBLIC)
        
        # Filter by category
        category_id = self.request.query_params.get('category_id')
        if category_id:
            queryset = queryset.filter(category_id=category_id)
            
        # Filter by search term
        search_term = self.request.query_params.get('q')
        if search_term:
            queryset = queryset.filter(name__icontains=search_term)
            
        return queryset

class SearchHistoryView(generics.ListCreateAPIView):
    """View to manage a user's search history."""
    serializer_class = SearchHistorySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Return last 5 search terms for the current user
        return SearchHistory.objects.filter(user=self.request.user)[:5]

    def perform_create(self, serializer):
        # Create or update the timestamp of the search term
        term = serializer.validated_data['term']
        obj, created = SearchHistory.objects.update_or_create(
            user=self.request.user, term=term,
            defaults={'timestamp': serializer.validated_data.get('timestamp')}
        )

    def delete(self, request, *args, **kwargs):
        # Delete all history for the user
        SearchHistory.objects.filter(user=self.request.user).delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class SearchHistoryDetailView(generics.DestroyAPIView):
    """View to delete a specific search history item."""
    serializer_class = SearchHistorySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return SearchHistory.objects.filter(user=self.request.user)