using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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

namespace Building
{
    /// <summary>
    /// CustomControl class of Building for construction menu.
    /// </summary>
    public class Building : Control
    {
        static Building()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(Building), new FrameworkPropertyMetadata(typeof(Building)));
        }

        // Type (Sawmill, Lumber, Hut, ...)
        public string Type
        {
            get => (string)GetValue(TypeProperty);
            set => SetValue(TypeProperty, value);
        }

        // Icon (Buildings/Camp.png, Buildings/Mine.png, ...)
        public string Icon
        {
            get => (string)GetValue(IconProperty);
            set => SetValue(IconProperty, value);
        }

        public static readonly DependencyProperty TypeProperty =
            DependencyProperty.Register("Type", typeof(string), typeof(Building));
        public static readonly DependencyProperty IconProperty =
            DependencyProperty.Register("Icon", typeof(string), typeof(Building));
    }
}
