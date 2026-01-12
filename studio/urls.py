from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import KnowledgeArtifactViewSet

router = DefaultRouter()
router.register(r'artifacts', KnowledgeArtifactViewSet, basename='knowledgeartifact')

urlpatterns = [
    path('', include(router.urls)),
]
