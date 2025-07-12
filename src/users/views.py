from django.shortcuts import redirect, render

def login_view(request):
    return render(request, 'views/login.html')