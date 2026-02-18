using Game.Support;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Game.ViewModels
{
    /// <summary>
    /// Implements the Building logic.
    /// </summary>
    public class BuildingViewModel : ViewModelBase
    {
        public BuildingViewModel(MainViewModel mainViewModel, string type)
        {
            _mainViewModel = mainViewModel; // Reference to MainViewModel
            Type = type;
            Icon = "Buildings/" + Type;

            // Populate resource cost dictionary
            _cost.Add("Wood", 0);
            _cost.Add("Stone", 0);
            _cost.Add("Food", 0);
            _cost.Add("Iron", 0);
            _cost.Add("Gold", 0);
            setCost();
        }

        private MainViewModel _mainViewModel;
        private RelayCommand _buildingClickCommand;
        private string _type;
        private string _icon;
        private Dictionary<string, int> _cost = new Dictionary<string, int>();

        // Type (Camp, Lumber, Tower, ...)
        public string Type
        {
            get => _type;
            set => SetProperty(ref _type, value);
        }

        // Icon (Buildings/Mine.png, ...)
        public string Icon
        {
            get => _icon;
            set => SetProperty(ref _icon, value);
        }

        // Set cost dictionary based on Building type
        private void setCost()
        {
            switch (Type)
            {
                case "Sawmill":
                    _cost["Wood"] = 5;
                    break;
                case "Lumber":
                    _cost["Wood"] = 10;
                    _cost["Food"] = 1;
                    break;
                case "Camp":
                    _cost["Wood"] = 8;
                    break;
                case "Quarry":
                    _cost["Wood"] = 30;
                    _cost["Food"] = 8;
                    _cost["Gold"] = 5;
                    break;
                case "Hut":
                    _cost["Wood"] = 20;
                    break;
                case "Windmill":
                    _cost["Wood"] = 25;
                    _cost["Gold"] = 5;
                    _cost["Stone"] = 1;
                    break;
                case "Tower":
                    _cost["Wood"] = 25;
                    _cost["Stone"] = 15;
                    break;
                case "Mine":
                    _cost["Wood"] = 20;
                    _cost["Stone"] = 20;
                    _cost["Food"] = 20;
                    _cost["Gold"] = 20;
                    break;
            }
        }

        // Invoked when clicked on the Building
        public RelayCommand BuildingClickCommand
        {
            get { return _buildingClickCommand ?? (_buildingClickCommand = new RelayCommand(ClickBuilding, ClickBuildingExecute)); }
        }

        // Process the Command in MainViewModel
        private void ClickBuilding(object obj)
        {
            _mainViewModel.ClickBuilding(_type, _cost);
        }

        private bool ClickBuildingExecute(object obj)
        {
            return true;
        }
    }
}
