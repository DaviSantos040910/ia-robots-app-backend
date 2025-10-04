# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Main admin site
    path('admin/', admin.site.urls),

    # API URLs
    # Include all URLs from the 'accounts' app under '/auth/'
    path('auth/', include('accounts.urls')),
    
    # Include all URLs from the 'bots' app under '/api/v1/bots/'
    # This single line replaces all the individual bot and admin paths
    path('api/v1/bots/', include('bots.urls')),
    
    # Include all URLs from the 'chat' app under '/api/v1/chats/'
    path('api/v1/chats/', include('chat.urls')),
    
    # Include all URLs from the 'explore' app under '/api/v1/explore/'
    path('api/v1/explore/', include('explore.urls')),
]