using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PredictiveModels.Tools.Commons.Utils;

namespace PredictiveModels.Tools.Compute.SVM
{
    public class SVCTrainParam : INotifyPropertyChanged
    {
        private bool _flattened = true;

        public event PropertyChangedEventHandler PropertyChanged;

        protected void NotifyPropertyChanged(string name)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(name));
            }
        }

        public bool FlattenData
        {
            get { return _flattened; }
            set
            {
                if (_flattened != value)
                {
                    _flattened = value;
                    NotifyPropertyChanged("FlattenData");
                }
            }
        }

        public void Copy(SVCTrainParam rhs)
        {
            _flattened = rhs._flattened;
        }

        public SVCTrainParam Clone()
        {
            SVCTrainParam clone = new SVCTrainParam();
            clone.Copy(this);
            return clone;
        }

        internal Dictionary<string, object> Properties()
        {
            Dictionary<string, object> properties = new Dictionary<string, object>();

            return properties;
        }
    }
}
