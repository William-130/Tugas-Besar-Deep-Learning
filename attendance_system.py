import pandas as pd
import os
from datetime import datetime
import json

class AttendanceSystem:
    def __init__(self, log_file='attendance_log.csv'):
        self.log_file = log_file
        self.init_log_file()
    
    def init_log_file(self):
        """Initialize attendance log file if not exists"""
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                'timestamp', 
                'date', 
                'time', 
                'name', 
                'confidence',
                'status'
            ])
            df.to_csv(self.log_file, index=False)
    
    def record_attendance(self, name, confidence, status='present'):
        """Record attendance entry"""
        now = datetime.now()
        
        entry = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'name': name,
            'confidence': round(confidence * 100, 2),
            'status': status
        }
        
        # Append to CSV
        df = pd.DataFrame([entry])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        
        return entry
    
    def get_attendance_log(self, date=None, name=None):
        """Get attendance log with optional filters"""
        try:
            df = pd.read_csv(self.log_file)
            
            if df.empty:
                return df
            
            # Filter by date
            if date:
                df = df[df['date'] == date]
            
            # Filter by name
            if name:
                df = df[df['name'] == name]
            
            return df
        except Exception as e:
            print(f"Error reading log: {e}")
            return pd.DataFrame()
    
    def get_daily_summary(self, date=None):
        """Get daily attendance summary"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        df = self.get_attendance_log(date=date)
        
        if df.empty:
            return {
                'date': date,
                'total_records': 0,
                'unique_people': 0,
                'people_list': []
            }
        
        # Get unique people for the day
        unique_people = df.groupby('name').agg({
            'timestamp': 'first',  # First detection time
            'confidence': 'mean'    # Average confidence
        }).reset_index()
        
        unique_people.columns = ['name', 'first_seen', 'avg_confidence']
        unique_people = unique_people.sort_values('first_seen')
        
        return {
            'date': date,
            'total_records': len(df),
            'unique_people': len(unique_people),
            'people_list': unique_people.to_dict('records')
        }
    
    def get_person_history(self, name, days=7):
        """Get attendance history for a specific person"""
        df = self.get_attendance_log(name=name)
        
        if df.empty:
            return pd.DataFrame()
        
        # Get last N days
        df['date'] = pd.to_datetime(df['date'])
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff_date]
        
        # Group by date
        daily_summary = df.groupby('date').agg({
            'timestamp': 'first',
            'confidence': 'mean',
            'name': 'count'
        }).reset_index()
        
        daily_summary.columns = ['date', 'first_seen', 'avg_confidence', 'detection_count']
        
        return daily_summary
    
    def get_all_people_summary(self, date=None):
        """Get summary of all people"""
        df = self.get_attendance_log(date=date)
        
        if df.empty:
            return pd.DataFrame()
        
        summary = df.groupby('name').agg({
            'timestamp': ['first', 'count'],
            'confidence': 'mean'
        }).reset_index()
        
        summary.columns = ['name', 'first_seen', 'total_detections', 'avg_confidence']
        summary = summary.sort_values('first_seen')
        
        return summary
    
    def export_to_csv(self, output_file=None, date=None):
        """Export attendance data to CSV"""
        if output_file is None:
            output_file = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = self.get_attendance_log(date=date)
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def clear_old_records(self, days=30):
        """Delete records older than specified days"""
        df = pd.read_csv(self.log_file)
        df['date'] = pd.to_datetime(df['date'])
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff_date]
        
        df.to_csv(self.log_file, index=False)
        
        return len(df)

if __name__ == "__main__":
    # Test the attendance system
    attendance = AttendanceSystem()
    
    # Record some test entries
    attendance.record_attendance("William Chan", 0.95)
    attendance.record_attendance("Abraham Ganda Napitu", 0.88)
    
    # Get daily summary
    summary = attendance.get_daily_summary()
    print("\nDaily Summary:")
    print(json.dumps(summary, indent=2))
    
    # Get all people summary
    print("\nAll People Summary:")
    print(attendance.get_all_people_summary())
