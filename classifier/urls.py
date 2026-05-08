from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name='home'),
    path('index/', views.index_view, name='index'),
    path("predict/", views.predict_view, name="predict"),
    path("about/", views.about_view, name="about"),
    path('upload/', views.upload_predict_view, name='upload_predict'),
    path('results/', views.results_view, name='results'),
    path("chatbot/", views.chatbot_view, name="chatbot"),
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]
