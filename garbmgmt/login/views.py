from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate , login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import re
from .models import *


def home(request):
    return render(request, 'home.html')


def user_register(request):
     if request.method == "POST":
        profile = request.FILES.get('profile')
        fullname = request.POST.get('fullname')
        phone = request.POST.get('phone')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        if password1 != password2:
            return redirect('user_register')

        result = User.objects.create(profile=profile, phone=phone, fullname=fullname, 	email=email, password=make_password(password2))
        result.save()

     return render(request,'user_register.html' )

def user_login(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(email=email, password=password)  # or email=... if custom model

        if user is not None:
            login(request, user)
            return redirect('user_dashboard')   # URL name, not file name

        else:
            messages.error(request, "Invalid email or password")
    
    return render(request, 'user_login.html')

def auth_login(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(email=email, password=password)

        if user is not None and user.role == "authority":
            login(request, user)
            return redirect('auth_dashboard')   # change URL name as needed

        else:
            messages.error(request, "Invalid authority credentials")

    return render(request, 'auth_login.html')


@login_required(login_url='user_login')
def user_dashboard(request):
    return render(request,'user_dashboard.html')


def user_logout(request):
    logout(request)
    return redirect('user_login')   # redirect to login page




def auth_dashboard(request):
    return render(request,'auth_dashboard.html')

