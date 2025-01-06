from django.urls import path
from . import views

urlpatterns = [
    path('', views.accueil, name='accueil'),  # Accueil
    path('upload/', views.upload_file, name='upload_file'),  # Upload de fichiers
    path('statistics/', views.perform_statistics, name='perform_statistics'),  # Vue pour les statistiques
    path('laws/', views.perform_laws, name='perform_laws'),  # Vue pour les lois
    path('visualize/', views.visualize, name='visualize'),  # Vue pour la visualisation
    path('tests/', views.perform_tests, name='perform_tests'),  # Vue pour les tests statistiques
   
    ]
