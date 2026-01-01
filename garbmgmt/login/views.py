from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth.hashers import make_password,check_password
from django.contrib.auth import authenticate , login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import re
from django.views.decorators.csrf import csrf_exempt
from .models import *
from django.http import JsonResponse
from .chatbot import get_response
import json
from .models import LegalDumpingLocation
from django.views.decorators.http import require_http_methods


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

        result = Normal_user.objects.create(profile=profile, phone=phone, fullname=fullname, 	email=email, password=make_password(password2))
        result.save()

     return render(request,'user_register.html' )




def user_login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        try:
            user = Normal_user.objects.get(email=email)

            # check hashed password
            if check_password(password, user.password):
                request.session["normal_user_id"] = user.id
                return redirect("user_dashboard")
            else:
                messages.error(request, "Invalid password")
                return redirect("user_login")

        except Normal_user.DoesNotExist:
            messages.error(request, "User not found")
            return redirect("user_login")

    return render(request, "user_login.html")


def auth_login(request):
    if request.method == "POST":
        auth_id = request.POST.get("auth_id")
        password = request.POST.get("password")

        try:
            user = Authority_user.objects.get(auth_id=auth_id)

            if user.password == password:  # direct comparison
                request.session['authority_user_id'] = user.id
                messages.success(request, "Authority Login Successful!")
                return redirect("auth_dashboard")
            else:
                messages.error(request, "Invalid password!")
                return redirect("auth_login")

        except Authority_user.DoesNotExist:
            messages.error(request, "Invalid Authority ID!")
            return redirect("auth_login")

    return render(request, "auth_login.html")


def user_dashboard(request):
    if 'normal_user_id' not in request.session:
        return redirect('user_login')
    return render(request, 'user_dashboard.html')


@csrf_exempt
def chatbot_api(request):
    print("VIEW HIT ✔")                     # Debug
    print("POST:", request.POST)            # Debug

    if request.method == "POST":
        user_message = request.POST.get("message", "")
        print("USER MESSAGE:", user_message)  # Debug

        reply = get_response(user_message)
        return JsonResponse({"reply": reply})

    return JsonResponse({"error": "Invalid request"}, status=400)

def user_logout(request):
    if 'normal_user_id' in request.session:
        del request.session['normal_user_id']
    return redirect('user_login')
  # redirect to login page

def auth_logout(request):
    request.session.flush()
    messages.success(request, "Logged out successfully.")
    return redirect('home')


def auth_dashboard(request):
    if 'authority_user_id' not in request.session:
        return redirect('auth_login')
    return render(request,'auth_dashboard.html')



@csrf_exempt  # (later you can do CSRF properly)
def save_location(request):

    authority_id = request.session.get("authority_user_id")
    if not authority_id:
        return JsonResponse({"error": "Not logged in"}, status=401)

    authority = Authority_user.objects.get(id=authority_id)

    if request.method == "POST":
        data = json.loads(request.body)

        location = LegalDumpingLocation.objects.create(
            name=data.get("name"),
            location_type=data.get("type"),
            latitude=data.get("lat"),
            longitude=data.get("lng"),
            added_by=authority
        )

        return JsonResponse({"status": "success"})

    return JsonResponse({"error": "Invalid request"}, status=400)




def get_locations(request):
    locations = LegalDumpingLocation.objects.all()

    data = []
    for loc in locations:
        data.append({
            "id": loc.id,
            "name": loc.name,
            "type": loc.location_type,
            "lat": loc.latitude,
            "lng": loc.longitude
        })

    return JsonResponse(data, safe=False)


@csrf_exempt
def delete_location(request):
    if request.method == "POST":
        data = json.loads(request.body)
        location_id = data.get("id")

        authority_id = request.session.get("authority_user_id")
        if not authority_id:
            return JsonResponse({"status": "unauthorized"}, status=403)

        location = get_object_or_404(
            LegalDumpingLocation,
            id=location_id,
            added_by_id=authority_id  # 🔐 authority ownership check
        )

        location.delete()  # 🔥 HARD DELETE
        return JsonResponse({"status": "success"})

    return JsonResponse({"status": "invalid"}, status=400)
