import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import RegisterForm, ImageUploadForm
from .models import UploadedImage


from torchvision.models import MobileNet_V2_Weights


model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()

imagenet_labels = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray",
    "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee", "water ouzel", "kite",
    "bald eagle", "vulture", "great grey owl", "cat", "dog", "fox", "bear", "lion", "tiger"
]


def preprocess_image(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def home(request):
    return render(request, 'home.html')


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'registration.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})


# Выход из системы
def logout_view(request):
    logout(request)
    return redirect('home')


# Панель управления и загрузка изображений
@login_required
def dashboard(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save(commit=False)
            image_instance.user = request.user
            image_instance.save()

            # Полный путь к загруженному изображению
            image_path = default_storage.path(image_instance.image.name)

            # Предобработка изображения и выполнение классификации
            input_batch = preprocess_image(image_path)

            # Проверка доступности CUDA (GPU)
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            # Выполнение классификации
            with torch.no_grad():
                output = model(input_batch)

            # Получение индекса класса с наибольшей вероятностью
            _, predicted_idx = torch.max(output, 1)

            # Получение имени класса из списка меток
            label = imagenet_labels[predicted_idx.item() % len(imagenet_labels)]
            image_instance.result = label
            image_instance.save()

            return redirect('dashboard')
    else:
        form = ImageUploadForm()

    images = UploadedImage.objects.filter(user=request.user)
    return render(request, 'dashboard.html', {'form': form, 'images': images})
