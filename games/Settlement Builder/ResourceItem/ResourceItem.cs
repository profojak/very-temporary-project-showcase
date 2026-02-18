using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ResourceItem
{
    /// <summary>
    /// CustomControl class of ResourceItem for resources menu.
    /// </summary>
    public class ResourceItem : Control
    {
        static ResourceItem()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(ResourceItem), new FrameworkPropertyMetadata(typeof(ResourceItem)));
        }

        // Type (Food, Gold, ...)
        public string Type
        {
            get => (string)GetValue(TypeProperty);
            set => SetValue(TypeProperty, value);
        }

        // Icon (ResourceItems/StoneIcon.png, ResourceItems/IronIcon.png, ...)
        public string Icon
        {
            get => (string)GetValue(IconProperty);
            set => SetValue(IconProperty, value);
        }

        // Count (negative value is game over)
        public int Count
        {
            get => (int)GetValue(CountProperty);
            set => SetValue(CountProperty, value);
        }

        public static readonly DependencyProperty TypeProperty = 
            DependencyProperty.Register("Type", typeof(string), typeof(ResourceItem));
        public static readonly DependencyProperty IconProperty =
            DependencyProperty.Register("Icon", typeof(string), typeof(ResourceItem));
        public static readonly DependencyProperty CountProperty =
            DependencyProperty.Register("Count", typeof(string), typeof(ResourceItem));
    }
}
