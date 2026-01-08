from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager 
from django.conf import settings

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
# models.py
class LegalDumpingLocation(models.Model):
    name = models.CharField(max_length=200)
    location_type = models.CharField(
        max_length=50, null=True, blank=True,
        choices=[
            ('bin', 'Garbage Bin'),
            ('recycling', 'Recycling Center'),
            ('transfer', 'Transfer Station'),
            ('hazardous', 'Hazardous Waste'),
        ]
    )
    latitude = models.FloatField()
    longitude = models.FloatField()
    is_active = models.BooleanField(default=True)
    added_by = models.ForeignKey(Authority_user, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)




class Camera(models.Model):
    camera_id = models.CharField(max_length=50, unique=True)
    location = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.camera_id} ({self.location})"



class DumpingEvent(models.Model):
    event_id = models.CharField(max_length=50, unique=True)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    actor = models.CharField(max_length=50, blank=True)
    dumping_video = models.FileField(upload_to="evidence/%Y/%m/%d/dumping/")
    
    # Add this line for location
    location = models.ForeignKey(
        LegalDumpingLocation,  # replace with your location model
        on_delete=models.CASCADE,
        null=True,             # optional, if some events may not have a location
        blank=True
    )
    
    def __str__(self):
        return f"{self.event_id} - {self.camera.camera_id}"


class NumberPlate(models.Model):
    event = models.ForeignKey(DumpingEvent, on_delete=models.CASCADE, related_name="plates")
    image = models.ImageField(upload_to="evidence/%Y/%m/%d/plates/")
    timestamp = models.DateTimeField(auto_now_add=True)
    plate_text = models.CharField(max_length=20, blank=True)  # optional if you want to store OCR results

    def __str__(self):
        return f"Plate {self.plate_text} for {self.event.event_id}"



#normal user garbage reporting
class GarbageReport(models.Model):
    user = models.ForeignKey(
        Normal_user,
        on_delete=models.CASCADE,
        related_name="reports"
    )
    location = models.CharField(max_length=255)
    description = models.TextField()
    severity = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)


#normal user garbage report storing
class GarbageEvidence(models.Model):
    report = models.ForeignKey(
        GarbageReport,
        on_delete=models.CASCADE,
        related_name="evidences"
    )
    file = models.FileField(upload_to="user_reports/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Evidence for Report #{self.report.id}"
