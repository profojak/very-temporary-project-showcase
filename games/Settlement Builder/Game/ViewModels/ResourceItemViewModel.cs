using Game.Support;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Game.ViewModels
{
    /// <summary>
    /// Implements the ResourceItem logic.
    /// </summary>
    public class ResourceItemViewModel : ViewModelBase
    {
        public ResourceItemViewModel(MainViewModel mainViewModel, string type)
        {
            _mainViewModel = mainViewModel; // Reference to MainViewModel
            Type = type;
            Icon = String.Format("ResourceItems/{0}Icon", Type);
            Count = 0;
        }

        private MainViewModel _mainViewModel;
        private RelayCommand _resourceItemClickCommand;
        private string _type;
        private string _icon;
        private int _count;

        // Type (Food, Gold, Iron, ...)
        public string Type
        {
            get => _type;
            set => SetProperty(ref _type, value);
        }

        // Icon (ResourceItems/GoldIcon.png, ...)
        public string Icon
        {
            get => _icon;
            set => SetProperty(ref _icon, value);
        }

        // Count
        public int Count
        {
            get => _count;
            set => SetProperty(ref _count, value);
        }

        // Add resource
        public void add(int count)
        {
            Count += count;
        }

        // Remove resource
        public void remove(int count)
        {
            Count -= count;
        }

        // Invoked when clicked on the ResourceItem
        public RelayCommand ResourceItemClickCommand
        {
            get { return _resourceItemClickCommand ?? (_resourceItemClickCommand = new RelayCommand(ClickResourceItem, ClickResourceItemExecute)); }
        }

        // Process the Command in MainViewModel
        private void ClickResourceItem(object obj)
        {
            _mainViewModel.ClickResourceItem(_type);
        }

        private bool ClickResourceItemExecute(object obj)
        {
            return true;
        }
    }
}
