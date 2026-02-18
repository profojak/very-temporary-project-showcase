using Game.Support;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Game.ViewModels
{
    /// <summary>
    /// Implements the Tile logic.
    /// </summary>
    public class TileViewModel : ViewModelBase
    {
        public TileViewModel(MainViewModel mainViewModel, string type, int seed, int index)
        {
            _mainViewModel = mainViewModel; // Reference to MainViewModel
            Type = type;
            Icon = String.Format("Tiles/{0}{1}", Type, seed.ToString());
            Index = index;
            ProductionType = "Wood";
            ProductionCount = 0;
        }

        private MainViewModel _mainViewModel;
        private RelayCommand _tileClickCommand;
        private string _type;
        private string _icon;
        private int _index;

        private string _production_type;
        private int _produciton_count;
        private string _cost_type;
        private int _cost_count;

        // Type (Empty, Sawmill, Hut, ...)
        public string Type
        {
            get => _type;
            set => SetProperty(ref _type, value);
        }

        // Icon (Tiles/Lumber1.png, ...)
        public string Icon
        {
            get => _icon;
            set => SetProperty(ref _icon, value);
        }

        // Index of the Tile on the game board
        public int Index
        {
            get => _index;
            set => SetProperty(ref _index, value);
        }

        // Which resource this Tile produces (Wood, Food, Stone, ...)
        public string ProductionType
        {
            get => _production_type;
            set => SetProperty(ref _production_type, value);
        }

        // How much this Tile produces
        public int ProductionCount
        {
            get => _produciton_count;
            set => SetProperty(ref _produciton_count, value);
        }

        // Which resource this Tile consumes (Wood, Food, Stone, ...)
        public string CostType
        {
            get => _cost_type;
            set => SetProperty(ref _cost_type, value);
        }

        // How much this Tile consumes
        public int CostCount
        {
            get => _cost_count;
            set => SetProperty(ref _cost_count, value);
        }

        // Update this Tile when construction happens
        public void update(string type, string production_type, int production_count,
                                        string cost_type, int cost_count)
        {
            Type = type;
            Icon = "Tiles/" + type + "1";
            ProductionType = production_type;
            ProductionCount = production_count;
            CostType = cost_type;
            CostCount = cost_count;
        }

        // Invoked when clicked on the Tile
        public RelayCommand TileClickCommand
        {
            get { return _tileClickCommand ?? (_tileClickCommand = new RelayCommand(ClickTile, ClickTileExecute)); }
        }

        // Process the Command in MainViewModel
        private void ClickTile(object obj)
        {
            _mainViewModel.ClickTile(_type, _index);
        }

        private bool ClickTileExecute(object obj)
        {
            return true;
        }
    }
}
