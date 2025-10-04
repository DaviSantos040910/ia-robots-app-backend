# chat/urls.py
from django.urls import path
from .views import ChatListCreateView, ChatMessageListCreateView, ChatBootstrapView

urlpatterns = [
    path('', ChatListCreateView.as_view(), name='chat-list-create'),
    # Note que o parâmetro é `bot_id`, não `chat_id`
    path('bootstrap/bot/<int:bot_id>/', ChatBootstrapView.as_view(), name='chat-bootstrap'),
    path('<int:chat_pk>/messages/', ChatMessageListCreateView.as_view(), name='chat-message-list-create'),
]