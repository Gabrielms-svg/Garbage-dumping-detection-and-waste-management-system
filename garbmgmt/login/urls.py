from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('',views.home,name="home"),
    path('user_register/',views.user_register,name="user_register"),
    path('user_login/',views.user_login,name="user_login"),
    path('auth_login/',views.auth_login,name='auth_login'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('auth_dashboard/',views.auth_dashboard,name="auth_dashboard"),
    path('user_logout/', views.user_logout, name='user_logout'),
    path('auth_logout',views.auth_logout,name='auth_logout'),
    path('chatbot/', views.chatbot_api, name='chatbot_api'),
    path("auth_dashboard/add-legal-location/", views.add_legal_location),
    

    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
