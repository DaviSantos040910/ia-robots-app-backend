# chat/urls.py
from django.urls import path
from .views import (
    ActiveChatListView, # Renamed from ChatListCreateView
    ChatMessageListView, 
    ChatBootstrapView,
    ArchiveChatView,      # New import
    ArchivedChatListView, # New import
    SetActiveChatView,
    ChatMessageAttachmentView 
)

urlpatterns = [
    # GET /api/v1/chats/ -> Lists active chats for the main screen
    path('', ActiveChatListView.as_view(), name='active-chat-list'),
    
    # GET /api/v1/chats/archived/bot/<bot_id>/ -> Lists archived chats for a bot
    path('archived/bot/<int:bot_id>/', ArchivedChatListView.as_view(), name='archived-chat-list'),
    
    # POST /api/v1/chats/<chat_id>/archive/ -> Archives a chat and creates a new one
    path('<int:chat_id>/archive/', ArchiveChatView.as_view(), name='archive-chat'),
    
    path('<int:chat_id>/set-active/', SetActiveChatView.as_view(), name='set-active-chat'),

    path('bootstrap/bot/<int:bot_id>/', ChatBootstrapView.as_view(), name='chat-bootstrap'),
    path('<int:chat_pk>/messages/', ChatMessageListView.as_view(), name='chat-message-list-create'),
    path('<int:chat_pk>/messages/attach/', ChatMessageAttachmentView.as_view(), name='chat-message-attach'),
]