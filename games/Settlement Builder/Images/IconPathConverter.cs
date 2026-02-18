using System;
using System.Globalization;
using System.Reflection;
using System.Windows.Data;
using System.Windows.Media.Imaging;

namespace Images
{
    /// <summary>
    /// Converts relative path (Tiles/Empty1.png) to a bitmap image.
    /// </summary>
    public class IconPathConverter : IValueConverter
    {
        // Acquire the actual image path
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var relativeFilePath = String.Format("pack://application:,,,/{0};component/Images/{1}.png",
                Assembly.GetExecutingAssembly().GetName().Name, value);

            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(relativeFilePath);
            bitmap.EndInit();

            return bitmap;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
