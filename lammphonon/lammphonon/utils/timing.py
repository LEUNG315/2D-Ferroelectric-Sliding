#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time analysis utilities for performance profiling
"""

import time
import datetime

def timestamp():
    """Return a timestamp string in format YYYYMMDD_HHMMSS"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def time_to_string(seconds):
    """Convert time in seconds to human-readable string"""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes} m {secs:.2f} s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = seconds % 60
        return f"{hours} h {minutes} m {secs:.2f} s"

class TimeProfiler:
    """Advanced time profiler for tracking execution time of multiple code sections"""
    
    def __init__(self, name="Profiler"):
        """Initialize profiler with a name"""
        self.name = name
        self.sections = {}
        self.stack = []
        self.current_section = None
        self.total_time = 0
    
    def start(self, section_name):
        """Start timing a section"""
        # Stop current section if any
        if self.current_section:
            self.stop()
        
        # Create section if it doesn't exist
        if section_name not in self.sections:
            self.sections[section_name] = {'total': 0.0, 'calls': 0, 'start': None}
        
        # Start timing
        self.sections[section_name]['start'] = time.time()
        self.sections[section_name]['calls'] += 1
        self.current_section = section_name
        self.stack.append(section_name)
        
        return self
    
    def stop(self):
        """Stop timing the current section"""
        if not self.current_section:
            return self
            
        # Calculate elapsed time
        elapsed = time.time() - self.sections[self.current_section]['start']
        self.sections[self.current_section]['total'] += elapsed
        self.sections[self.current_section]['start'] = None
        
        # Pop from stack
        if self.stack:
            self.stack.pop()
            
        # Set new current section
        if self.stack:
            self.current_section = self.stack[-1]
        else:
            self.current_section = None
            
        return self
    
    def summary(self):
        """Return a summary of profiling results"""
        if self.current_section:
            self.stop()
            
        # Calculate total time
        total_time = sum(section['total'] for section in self.sections.values())
        self.total_time = total_time
        
        # Build summary string
        summary = f"Time Profiling Summary for '{self.name}':\n"
        summary += "-" * 80 + "\n"
        summary += f"{'Section':<30} {'Time':<15} {'Calls':<10} {'Avg Time':<15} {'Percentage':<10}\n"
        summary += "-" * 80 + "\n"
        
        # Sort sections by total time (descending)
        sorted_sections = sorted(self.sections.items(), 
                                key=lambda x: x[1]['total'], 
                                reverse=True)
        
        # Add section details
        for name, stats in sorted_sections:
            avg_time = stats['total'] / stats['calls'] if stats['calls'] > 0 else 0
            percentage = (stats['total'] / total_time * 100) if total_time > 0 else 0
            summary += f"{name:<30} {time_to_string(stats['total']):<15} {stats['calls']:<10} {time_to_string(avg_time):<15} {percentage:.2f}%\n"
            
        summary += "-" * 80 + "\n"
        summary += f"Total Time: {time_to_string(total_time)}\n"
        
        return summary

class Timer:
    """Simple timer for measuring execution time"""
    
    def __init__(self, name=""):
        """Initialize timer with optional name"""
        self.name = name
        self.start_time = None
        self.stop_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer"""
        self.stop_time = time.time()
        return self
    
    def elapsed(self):
        """Return elapsed time in seconds"""
        if self.start_time is None:
            return 0
        end_time = self.stop_time if self.stop_time else time.time()
        return end_time - self.start_time
    
    def elapsed_str(self):
        """Return elapsed time as formatted string"""
        return time_to_string(self.elapsed())
    
    def __enter__(self):
        """Start timing when entering context"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Stop timing when exiting context"""
        self.stop()
        if self.name:
            print(f"{self.name}: {self.elapsed_str()}") 