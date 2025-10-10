from importlib import reload
from django.http import JsonResponse
from django.shortcuts import redirect, render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.mail import send_mail
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import LikedListing, Listing
from .forms import ListingForm
from users.forms import LocationForm
from .filters import ListingFilter
import numpy as np
import json
import os
import joblib
import xgboost as xgb
from django.conf import settings
from django.http import JsonResponse
import traceback


# Load artifacts once at the top (already done in your file)
model_dir = os.path.join(settings.BASE_DIR, 'main', 'ml')
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(model_dir, "best_xgb.model"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
model_te_mapping = joblib.load(os.path.join(model_dir, "model_te_mapping.pkl"))
brand_te_mapping = joblib.load(os.path.join(model_dir, "brand_te_mapping.pkl"))
onehot_columns = joblib.load(os.path.join(model_dir, "onehot_columns.pkl"))
global_mean = joblib.load(os.path.join(model_dir, "global_mean.pkl"))

@csrf_exempt
def predict_price(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            brand = data.get("brand")
            model = data.get("model")
            year = int(data.get("year"))
            mileage = float(data.get("mileage"))
            engine_capacity = float(data.get("engine_capacity"))
            transmission = data.get("transmission")  # e.g., "Manual"
            fuel_type = data.get("fuel_type")        # e.g., "Petrol"
            ownership = data.get("ownership")        # e.g., "2nd owner"
            spare_key = int(data.get("spare_key", 1))  # binary 0/1

            # Base numerical features
            age = 2025 - year
            model_encoded = model_te_mapping.get(model, global_mean)

            # Start with base dict
            input_dict = {
                'engine_capacity(CC)': engine_capacity,
                'km_driven': mileage,
                'spare_key': spare_key,
                'age': age,
                'model_te': model_encoded,
            }

            # One-hot encoding: default to 0 for all known values
            for col in onehot_columns:
                if col not in input_dict:
                    input_dict[col] = 0

            # Set the matching one-hot features to 1
            brand_col = f"brand_{brand}"
            if brand_col in input_dict:
                input_dict[brand_col] = 1

            transmission_col = f"transmission_{transmission}"
            if transmission_col in input_dict:
                input_dict[transmission_col] = 1

            fuel_col = f"fuel_type_{fuel_type}"
            if fuel_col in input_dict:
                input_dict[fuel_col] = 1

            ownership_col = f"ownership_{ownership}"
            if ownership_col in input_dict:
                input_dict[ownership_col] = 1

            # Final input row
            input_row = [input_dict[col] for col in onehot_columns]

            # Scale if needed (based on training)
            # Step 1: Create full input_dict (as you're doing).

            # Step 2: Build input_row with all 25 features
            input_row = [input_dict[col] for col in onehot_columns]  # full 25 features

            # Step 3: Build a separate DataFrame for the 3 numeric features
            num_cols = ['engine_capacity(CC)', 'km_driven', 'age']
            numeric_vals = [[input_dict[col] for col in num_cols]]
            scaled_numeric = scaler.transform(numeric_vals)[0]  # shape: (3,)

            # Step 4: Replace unscaled numeric features in input_row with scaled ones
            for idx, col in enumerate(onehot_columns):
                if col in num_cols:
                    input_row[idx] = scaled_numeric[num_cols.index(col)]

            # Step 5: Predict using model
            dmatrix = xgb.DMatrix([input_row], feature_names=onehot_columns)
            price = xgb_model.predict(dmatrix)[0]


            return JsonResponse({"success": True, "price": float(round(price, 2))})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Invalid request"})

def main_view(request):
    return render(request, "views/main.html", {"name": "AutoMax"})


@login_required
def home_view(request):
    listings = Listing.objects.all()
    listing_filter = ListingFilter(request.GET, queryset=listings)
    user_liked_listings = LikedListing.objects.filter(
        profile=request.user.profile).values_list('listing')
    liked_listings_ids = [l[0] for l in user_liked_listings]
    context = {
        'listing_filter': listing_filter,
        'liked_listings_ids': liked_listings_ids,
    }
    return render(request, "views/home.html", context)


@login_required
def list_view(request):
    if request.method == 'POST':
        try:
            listing_form = ListingForm(request.POST, request.FILES)
            location_form = LocationForm(request.POST)
            
            if listing_form.is_valid() and location_form.is_valid():
                listing = listing_form.save(commit=False)
                listing_location = location_form.save()
                listing.seller = request.user.profile
                listing.location = listing_location
                listing.save()
                
                messages.success(
                    request, f'{listing.model} Listing Posted Successfully!')
                return redirect('home')
            else:
                print("Form errors:", listing_form.errors, location_form.errors)
                messages.error(
                    request, 'Invalid form data. Please check and try again.')
        except Exception as e:
            print("Error while posting listing:", e)
            messages.error(
                request, 'An unexpected error occurred while posting the listing.')
    else:  # GET request
        listing_form = ListingForm()
        location_form = LocationForm()
        
    return render(request, 'views/list.html', {
        'listing_form': listing_form,
        'location_form': location_form,
    })



@login_required
def listing_view(request, id):
    try:
        listing = Listing.objects.get(id=id)
        if listing is None:
            raise Exception
        return render(request, 'views/listing.html', {'listing': listing, })
    except Exception as e:
        messages.error(request, f'Invalid UID {id} was provided for listing.')
        return redirect('home')


@login_required
def edit_view(request, id):
    try:
        listing = Listing.objects.get(id=id)
        if listing is None:
            raise Exception
        if request.method == 'POST':
            listing_form = ListingForm(
                request.POST, request.FILES, instance=listing)
            location_form = LocationForm(
                request.POST, instance=listing.location)
            if listing_form.is_valid and location_form.is_valid:
                listing_form.save()
                location_form.save()
                messages.info(request, f'Listing {id} updated successfully!')
                return redirect('home')
            else:
                messages.error(
                    request, f'An error occured while trying to edit the listing.')
                return reload()
        else:
            listing_form = ListingForm(instance=listing)
            location_form = LocationForm(instance=listing.location)
        context = {
            'location_form': location_form,
            'listing_form': listing_form
        }
        return render(request, 'views/edit.html', context)
    except Exception as e:
        messages.error(
            request, f'An error occured while trying to access the edit page.')
        return redirect('home')


@login_required
def like_listing_view(request, id):
    listing = get_object_or_404(Listing, id=id)

    liked_listing, created = LikedListing.objects.get_or_create(
        profile=request.user.profile, listing=listing)

    if not created:
        liked_listing.delete()
    else:
        liked_listing.save()

    return JsonResponse({
        'is_liked_by_user': created,
    })


@login_required
def inquire_listing_using_email(request, id):
    listing = get_object_or_404(Listing, id=id)
    try:
        emailSubject = f'{request.user.username} is interested in {listing.model}'
        emailMessage = (
            f'Hi {listing.seller.user.username},\n\n'
            f'{request.user.username} is interested in your {listing.model} listing on ResaleX.\n'
            f'Please log in to ResaleX to view details.\n\n'
            'Regards,\nResaleX Team'
        )
        send_mail(
            emailSubject,
            emailMessage,
            settings.DEFAULT_FROM_EMAIL,
            [listing.seller.user.email],
            fail_silently=False  # Change to True only in production
        )
        return JsonResponse({"success": True})
    except Exception as e:
        print(f"Email send error: {e}")  # Also log it for debugging
        return JsonResponse({
            "success": False,
            "info": str(e),  # Convert error to string
        })
