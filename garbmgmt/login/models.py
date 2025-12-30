from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager 

# Create your models here.

class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email field must be set")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self.create_user(email, password, **extra_fields)


# Custom user model
class User(AbstractUser):
    email = models.EmailField(unique=True)  

    username = None

    objects = CustomUserManager()

    class Meta:
        db_table = 'user'

    def _str_(self):
        return self.fullname

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

class Normal_user(models.Model):
    fullname = models.CharField(max_length=50)
    username = models.CharField(max_length=50)
    email = models.EmailField(max_length=100, unique=True)
    phone = models.CharField(max_length=15, unique=True)
    profile = models.ImageField(upload_to='profiles', null=True, blank=True)
    password = models.CharField(max_length=255)


    def _str_(self):
        return f"{self.first_name} {self.last_name}"
    
    class Meta:
        db_table = 'normal_user'  # Change table name here



class Authority_user(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    auth_id = models.CharField(max_length=25, unique=True)
    email = models.EmailField(max_length=100, unique=True)
    phone = models.CharField(max_length=15, unique=True)
    profile_image = models.ImageField(upload_to='auth_profiles/', null=True, blank=True)
    password = models.CharField(max_length=256) 
  

    def _str_(self):
        return f"{self.first_name} {self.last_name}"
    
    class Meta:
        db_table = 'authority_user'  # Change table name here



# models.py
class LegalDumpingLocation(models.Model):
    name = models.CharField(max_length=200)
    latitude = models.FloatField()
    longitude = models.FloatField()
    is_active = models.BooleanField(default=True)
    added_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

