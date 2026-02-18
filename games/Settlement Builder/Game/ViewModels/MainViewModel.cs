using Game.Support;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace Game.ViewModels
{
    /// <summary>
    /// Implements the game logic.
    /// </summary>
    public class MainViewModel : ViewModelBase
    {
        // ObservableCollections of CustomControls
        public ObservableCollection<ResourceItemViewModel> ResourceItems { get; set; } = new ObservableCollection<ResourceItemViewModel>();
        public ObservableCollection<TileViewModel> Tiles { get; set; } = new ObservableCollection<TileViewModel>();
        public ObservableCollection<BuildingViewModel> Buildings { get; set; } = new ObservableCollection<BuildingViewModel>();

        private Random _random;
        private Manual _manual;
        private RelayCommand _nextDayCommand;

        // Tiles generation variables
        private readonly int MAX_FOOD_TILES;
        private readonly int MAX_STONE_TILES;
        private readonly int MAX_IRON_TILES;
        private int _food_tiles_count = 0;
        private int _stone_tiles_count = 0;
        private int _iron_tiles_count = 0;

        private string _manual_icon = "Logo";
        private string _manual_label = "Welcome! Your goal is to build an Iron mine.\n" +
            "If you run out of resources, game over!\n" +
            "Click on any image to show a manual.";
        private string _days = "1";
        private int _days_ref = 1;
        private int _selected_tile = -1;

        public MainViewModel()
        {
            _random = new Random();
            _manual = new Manual();

            // Pseudorandom number of generated Food and Stone tiles
            MAX_FOOD_TILES = _random.Next(3, 6);
            MAX_STONE_TILES = _random.Next(2, 5);
            MAX_IRON_TILES = 1;

            generateTiles();
            addResource("Wood", 5); // Start with 5 Wood
        }

        // ManualIcon (changes based on the selected game element)
        public string ManualIcon
        {
            get => _manual_icon;
            set => SetProperty(ref _manual_icon, value);
        }

        // ManualLabel (acquired from Manual class)
        public string ManualLabel
        {
            get => _manual_label;
            set => SetProperty(ref _manual_label, value);
        }

        // Number of elapsed days
        public string Days
        {
            get => "Days: " + _days;
            set => SetProperty(ref _days, value);
        }

        // When player clicks on a Tile
        public void ClickTile(string type, int index)
        {
            ManualIcon = "Tiles/" + type + "1";
            ManualLabel = _manual.GetText("Tile" + type);
            _selected_tile = index; // Set Tile index

            Buildings.Clear(); // Clear the construction menu
            generateConstructionMenu(type); // Fill construction menu with corresponding Buildings
        }

        // When player clicks on a ResourceItem
        public void ClickResourceItem(string type)
        {
            ManualIcon = "ResourceItems/" + type + "Icon";
            ManualLabel = _manual.GetText("ResourceItem" + type);

            Buildings.Clear(); // Clear the construction menu
        }

        // When player clicks on a Building
        public void ClickBuilding(string type, Dictionary<string, int> cost)
        {
            ManualIcon = "Tiles/" + type + "1";
            ManualLabel = _manual.GetText("Tile" + type);

            if (checkResourceDict(cost)) // If player has enough resources to build
            {
                // Confirm the construction of a building
                if (MessageBox.Show("Do you want to build this building?", String.Empty,
                    MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.Yes)
                {
                    foreach (TileViewModel tile in Tiles)
                    {
                        if (tile.Index == _selected_tile) // Find the selected tile
                        {
                            switch (type)
                            {
                                case "Sawmill":
                                    tile.update(type, "Wood", 1, "Wood", 0); // Update tile
                                    removeResourceDict(cost); // Remove resources needed for construciton
                                    break;
                                case "Lumber":
                                    tile.update(type, "Wood", 3, "Food", 1);
                                    removeResourceDict(cost);
                                    break;
                                case "Camp":
                                    tile.update(type, "Food", 2, "Wood", 0);
                                    removeResourceDict(cost);
                                    break;
                                case "Quarry":
                                    tile.update(type, "Stone", 1, "Food", 5);
                                    removeResourceDict(cost);
                                    break;
                                case "Hut":
                                    tile.update(type, "Gold", 1, "Food", 2);
                                    removeResourceDict(cost);
                                    break;
                                case "Windmill":
                                    tile.update(type, "Food", 1, "Gold", 1);
                                    removeResourceDict(cost);
                                    break;
                                case "Tower":
                                    tile.update(type, "Food", 5, "Wood", 5);
                                    removeResourceDict(cost);
                                    break;
                                case "Mine":
                                    tile.update(type, "Iron", 1, "Wood", 0);
                                    removeResourceDict(cost);
                                    // Player won the game by building an Iron mine!
                                    MessageBox.Show("Congratulations! You beat the game!", String.Empty,
                                        MessageBoxButton.OK, MessageBoxImage.Exclamation);
                                    break;
                            }
                        }
                    }
                }
            }
            else
            {
                // Player does not have enough resources to build
                MessageBox.Show("Not enough resources!", String.Empty,
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // Generate tiles on game board when the game starts
        private void generateTiles()
        {
            ResourceItems.Add(new ResourceItemViewModel(this, "Wood"));
            ResourceItems.Add(new ResourceItemViewModel(this, "Stone"));
            ResourceItems.Add(new ResourceItemViewModel(this, "Iron"));
            ResourceItems.Add(new ResourceItemViewModel(this, "Food"));
            ResourceItems.Add(new ResourceItemViewModel(this, "Gold"));

            // Game board has 10 columns and 5 rows
            for (int i = 0; i < 10 * 5; i++)
            {
                // Pseudorandom generation of forrests
                if (_random.Next(11) < 4)
                {
                    Tiles.Add(new TileViewModel(this, "Wood", _random.Next(1, 4), i));
                }
                // Pseudorandom generation of wild boar herds
                else if ((_random.Next(13) < 3) && (_food_tiles_count < MAX_FOOD_TILES))
                {
                    Tiles.Add(new TileViewModel(this, "Food", _random.Next(1, 3), i));
                    _food_tiles_count++;
                }
                // Pseudorandom generation of rocks
                else if ((_random.Next(14) < 6) && (_stone_tiles_count < MAX_STONE_TILES))
                {
                    Tiles.Add(new TileViewModel(this, "Stone", _random.Next(1, 4), i));
                    _stone_tiles_count++;
                }
                // Pseudorandom generation of iron veins
                else if ((_random.Next(20) < 4) && (_iron_tiles_count < MAX_IRON_TILES))
                {
                    Tiles.Add(new TileViewModel(this, "Iron", 1, i));
                    _iron_tiles_count++;
                }
                // Create an empty tile otherwise
                else
                {
                    Tiles.Add(new TileViewModel(this, "Empty", _random.Next(1, 4), i));
                }
            }
        }


        // Generate construction menu based on the selected tile
        private void generateConstructionMenu(string type)
        {
            switch (type)
            {
                case "Wood":
                    Buildings.Add(new BuildingViewModel(this, "Sawmill"));
                    break;
                case "Sawmill":
                    Buildings.Add(new BuildingViewModel(this, "Lumber"));
                    break;
                case "Food":
                    Buildings.Add(new BuildingViewModel(this, "Camp"));
                    break;
                case "Stone":
                    Buildings.Add(new BuildingViewModel(this, "Quarry"));
                    break;
                case "Empty":
                    Buildings.Add(new BuildingViewModel(this, "Hut"));
                    Buildings.Add(new BuildingViewModel(this, "Windmill"));
                    break;
                case "Camp":
                    Buildings.Add(new BuildingViewModel(this, "Tower"));
                    break;
                case "Iron":
                    Buildings.Add(new BuildingViewModel(this, "Mine"));
                    break;
            }
        }

        // Add resource
        public void addResource(string type, int count)
        {
            foreach (ResourceItemViewModel resource in ResourceItems)
            {
                if (resource.Type == type)
                {
                    resource.add(count);
                }
            }
        }

        // Remove resource
        private void removeResource(string type, int count)
        {
            foreach (ResourceItemViewModel resource in ResourceItems)
            {
                if (resource.Type == type)
                {
                    resource.remove(count);
                }
            }
        }

        // Check for resources from a resource dictionary
        private bool checkResourceDict(Dictionary<string, int> cost)
        {
            foreach (ResourceItemViewModel resource in ResourceItems)
            {
                if (resource.Count < cost[resource.Type])
                {
                    return false;
                }
            }
            return true;
        }

        // Remove resources from a resource dictionary
        private void removeResourceDict(Dictionary<string, int> cost)
        {
            foreach (ResourceItemViewModel resource in ResourceItems)
            {
                resource.remove(cost[resource.Type]);
            }
        }

        // Invoked when clicked on the Next day! button
        public RelayCommand NextDayCommand
        {
            get { return _nextDayCommand ?? (_nextDayCommand = new RelayCommand(NextDay, NextDayExecute)); }
        }

        // Move to next day
        private void NextDay(object obj)
        {
            _days_ref++;
            Days = _days_ref.ToString();

            // Add and then remove resources each tile generates and consumes
            foreach (TileViewModel tile in Tiles)
            {
                addResource(tile.ProductionType, tile.ProductionCount);
                removeResource(tile.CostType, tile.CostCount);
            }

            // Check for game over
            foreach (ResourceItemViewModel resource in ResourceItems)
            {
                if (resource.Count < 0)
                {
                    MessageBox.Show("You ran out of resources! Game over!", String.Empty,
                        MessageBoxButton.OK, MessageBoxImage.Error);
                    System.Windows.Application.Current.Shutdown();
                }
            }
        }

        private bool NextDayExecute(object obj)
        {
            return true;
        }
    }
}
