using AracSayar.Models;

namespace AracSayar;

public partial class MeasurementsPage : ContentPage
{
    // Page constructor
    public MeasurementsPage()
    {
        InitializeComponent();
    }

    // Loads measurements when page appears
    protected override async void OnAppearing()
    {
        base.OnAppearing();

        var measurements = await App.Database.GetMeasurementsAsync();
        BindingContext = measurements;
    }

    // Deletes selected measurement
    private async void OnDeleteInvoked(object sender, EventArgs e)
    {
        var swipeItem = sender as SwipeItem;
        var measurement = swipeItem?.BindingContext as MeasurementResult;

        if (measurement == null)
            return;

        var confirm = await DisplayAlert(
            "Sil",
            "Bu ölçümü silmek istiyor musun?",
            "Evet",
            "Hayýr");

        if (!confirm)
            return;

        await App.Database.DeleteMeasurementAsync(measurement);

        var measurements = await App.Database.GetMeasurementsAsync();
        BindingContext = measurements;
    }
}
