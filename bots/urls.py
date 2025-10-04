# bots/urls.py
from django.urls import path
from .views import BotListCreateView, BotDetailView
from .admin_views import (
    AdminBotListView, AdminBotDetailView, AdminCategoryView,
    AdminCategoryDetailView, AdminSetUserPremiumView
)

urlpatterns = [
    # --- User-facing URLs ---
    # GET /api/v1/bots/ -> List user's bots
    # POST /api/v1/bots/ -> Create a new bot
    path('', BotListCreateView.as_view(), name='bot-list-create'),

    # GET, PUT, DELETE /api/v1/bots/<id>/ -> Bot details for the owner
    path('<int:pk>/', BotDetailView.as_view(), name='bot-detail'),

    # --- Admin URLs ---
    # These URLs are prefixed with '/api/v1/bots/' from the root urls.py
    # GET /api/v1/bots/admin/ -> List all non-private bots for admin
    path('admin/', AdminBotListView.as_view(), name='admin-bot-list'),

    # DELETE /api/v1/bots/admin/<id>/ -> Delete a bot
    path('admin/<int:pk>/', AdminBotDetailView.as_view(), name='admin-bot-detail'),

    # GET, POST /api/v1/bots/admin/categories/ -> List or create categories
    path('admin/categories/', AdminCategoryView.as_view(), name='admin-category-list-create'),

    # DELETE /api/v1/bots/admin/categories/<id>/ -> Delete a category
    path('admin/categories/<int:pk>/', AdminCategoryDetailView.as_view(), name='admin-category-detail'),

    # POST /api/v1/bots/admin/users/<user_id>/set-premium/ -> Toggle premium status
    path('admin/users/<int:user_id>/set-premium/', AdminSetUserPremiumView.as_view(), name='admin-set-premium'),
]