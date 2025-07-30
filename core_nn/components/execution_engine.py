"""
Edge-Efficient Modular Execution Engine - Asynchronous modular execution system.

The execution engine operates in asynchronous blocks, not in lockstep like transformers.
Components can be dynamically offloaded, compressed, or run on-demand.
"""

import torch
import torch.nn as nn
import asyncio
import threading
import queue
import time
import psutil
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from collections import defaultdict

from ..config.schema import ExecutionEngineConfig


class ModuleState(Enum):
    """Module execution states."""
    IDLE = "idle"
    LOADING = "loading"
    ACTIVE = "active"
    PROCESSING = "processing"
    OFFLOADING = "offloading"
    OFFLOADED = "offloaded"
    ERROR = "error"


@dataclass
class ModuleInfo:
    """Information about a module."""
    module_id: str
    module_name: str
    state: ModuleState
    priority: int
    memory_usage_mb: float
    last_access: float
    processing_time_ms: float
    error_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionTask:
    """Task for module execution."""
    task_id: str
    module_id: str
    input_data: torch.Tensor
    callback: Optional[Callable] = None
    priority: int = 0
    timeout: float = 30.0
    metadata: Optional[Dict[str, Any]] = None


class ModuleManager:
    """Manages individual modules and their states."""
    
    def __init__(self, config: ExecutionEngineConfig):
        self.config = config
        self.modules: Dict[str, nn.Module] = {}
        self.module_info: Dict[str, ModuleInfo] = {}
        self.offloaded_modules: Dict[str, bytes] = {}  # Serialized modules
        
        # Memory monitoring
        self.memory_budget_bytes = config.memory_budget_gb * 1024 * 1024 * 1024
        self.offload_threshold = config.offload_threshold
        
    def register_module(self, 
                       module_id: str, 
                       module: nn.Module, 
                       priority: int = 0) -> bool:
        """Register a module with the manager."""
        try:
            # Estimate memory usage
            memory_usage = self._estimate_memory_usage(module)
            
            # Check if we have space
            if self._get_current_memory_usage() + memory_usage > self.memory_budget_bytes:
                # Try to free space
                if not self._free_memory(memory_usage):
                    return False
            
            # Register module
            self.modules[module_id] = module
            self.module_info[module_id] = ModuleInfo(
                module_id=module_id,
                module_name=module.__class__.__name__,
                state=ModuleState.IDLE,
                priority=priority,
                memory_usage_mb=memory_usage / (1024 * 1024),
                last_access=time.time(),
                processing_time_ms=0.0
            )
            
            return True
            
        except Exception as e:
            print(f"Error registering module {module_id}: {e}")
            return False
    
    def get_module(self, module_id: str) -> Optional[nn.Module]:
        """Get module, loading from offload if necessary."""
        if module_id in self.modules:
            self.module_info[module_id].last_access = time.time()
            return self.modules[module_id]
        
        if module_id in self.offloaded_modules:
            # Load from offload
            return self._load_from_offload(module_id)
        
        return None
    
    def offload_module(self, module_id: str) -> bool:
        """Offload module to free memory."""
        if module_id not in self.modules:
            return False
        
        try:
            # Serialize module
            module = self.modules[module_id]
            serialized = torch.save(module.state_dict(), None)
            
            # Store serialized version
            self.offloaded_modules[module_id] = serialized
            
            # Remove from active memory
            del self.modules[module_id]
            self.module_info[module_id].state = ModuleState.OFFLOADED
            
            return True
            
        except Exception as e:
            print(f"Error offloading module {module_id}: {e}")
            self.module_info[module_id].state = ModuleState.ERROR
            return False
    
    def _load_from_offload(self, module_id: str) -> Optional[nn.Module]:
        """Load module from offloaded state."""
        if module_id not in self.offloaded_modules:
            return None
        
        try:
            # Check memory availability
            info = self.module_info[module_id]
            memory_needed = info.memory_usage_mb * 1024 * 1024
            
            if self._get_current_memory_usage() + memory_needed > self.memory_budget_bytes:
                if not self._free_memory(memory_needed):
                    return None
            
            # Deserialize module (simplified - would need module class info)
            # This is a placeholder - real implementation would need module reconstruction
            serialized_data = self.offloaded_modules[module_id]
            
            # For now, return None - real implementation would reconstruct module
            # module = ModuleClass()
            # module.load_state_dict(torch.load(serialized_data))
            
            # Update state
            info.state = ModuleState.ACTIVE
            info.last_access = time.time()
            
            # Remove from offload storage
            del self.offloaded_modules[module_id]
            
            return None  # Placeholder
            
        except Exception as e:
            print(f"Error loading module {module_id} from offload: {e}")
            self.module_info[module_id].state = ModuleState.ERROR
            return None
    
    def _estimate_memory_usage(self, module: nn.Module) -> int:
        """Estimate memory usage of a module."""
        total_params = sum(p.numel() for p in module.parameters())
        # Assume float32 parameters + some overhead
        return total_params * 4 * 2  # 2x for gradients and activations
    
    def _get_current_memory_usage(self) -> int:
        """Get current memory usage of active modules."""
        return sum(
            int(info.memory_usage_mb * 1024 * 1024)
            for info in self.module_info.values()
            if info.state in [ModuleState.ACTIVE, ModuleState.PROCESSING]
        )
    
    def _free_memory(self, needed_bytes: int) -> bool:
        """Free memory by offloading modules."""
        # Sort modules by priority and last access
        candidates = [
            (module_id, info) for module_id, info in self.module_info.items()
            if info.state in [ModuleState.IDLE, ModuleState.ACTIVE] and module_id in self.modules
        ]
        
        candidates.sort(key=lambda x: (x[1].priority, -x[1].last_access))
        
        freed_bytes = 0
        for module_id, info in candidates:
            if freed_bytes >= needed_bytes:
                break
            
            if self.offload_module(module_id):
                freed_bytes += int(info.memory_usage_mb * 1024 * 1024)
        
        return freed_bytes >= needed_bytes


class TaskScheduler:
    """Schedules and executes tasks across modules."""
    
    def __init__(self, config: ExecutionEngineConfig, module_manager: ModuleManager):
        self.config = config
        self.module_manager = module_manager
        
        # Task queues
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # Execution control
        self.max_concurrent = config.max_concurrent_modules
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.cpu_threads if config.cpu_threads > 0 else None
        )
        
        # Statistics
        self.task_stats = defaultdict(int)
        
    def submit_task(self, task: ExecutionTask) -> str:
        """Submit a task for execution."""
        # Add to queue with priority
        priority = -task.priority  # Negative for max-heap behavior
        self.task_queue.put((priority, time.time(), task))
        
        return task.task_id
    
    def execute_tasks(self) -> Dict[str, Any]:
        """Execute pending tasks."""
        results = {}
        
        while not self.task_queue.empty() and len(self.active_tasks) < self.max_concurrent:
            try:
                _, _, task = self.task_queue.get_nowait()
                
                # Get module
                module = self.module_manager.get_module(task.module_id)
                if module is None:
                    results[task.task_id] = {"error": f"Module {task.module_id} not available"}
                    continue
                
                # Execute task
                if self.config.async_execution:
                    future = self.thread_pool.submit(self._execute_task, task, module)
                    self.active_tasks[task.task_id] = task
                    results[task.task_id] = {"status": "submitted", "future": future}
                else:
                    result = self._execute_task(task, module)
                    results[task.task_id] = result
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error executing task: {e}")
        
        return results
    
    def _execute_task(self, task: ExecutionTask, module: nn.Module) -> Dict[str, Any]:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Update module state
            module_info = self.module_manager.module_info[task.module_id]
            module_info.state = ModuleState.PROCESSING
            
            # Execute module
            with torch.no_grad():
                output = module(task.input_data)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            module_info.processing_time_ms = processing_time
            module_info.state = ModuleState.ACTIVE
            module_info.last_access = time.time()
            
            # Call callback if provided
            if task.callback:
                task.callback(output)
            
            return {
                "status": "completed",
                "output": output,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            # Handle error
            module_info = self.module_manager.module_info[task.module_id]
            module_info.error_count += 1
            module_info.state = ModuleState.ERROR
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        
        finally:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]


class EdgeEfficientModularExecutionEngine(nn.Module):
    """
    Edge-Efficient Modular Execution Engine - Asynchronous modular execution.
    
    Operates in asynchronous blocks with dynamic component offloading,
    compression, and on-demand execution for edge devices.
    """
    
    def __init__(self, config: ExecutionEngineConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.module_manager = ModuleManager(config)
        self.task_scheduler = TaskScheduler(config, self.module_manager)
        
        # System monitoring
        self.system_monitor = SystemMonitor(config)
        
        # Execution statistics
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "memory_usage_history": [],
            "cpu_usage_history": []
        }
        
    def register_component(self, 
                          component_id: str, 
                          component: nn.Module, 
                          priority: int = 0) -> bool:
        """Register a component with the execution engine."""
        return self.module_manager.register_module(component_id, component, priority)
    
    def execute_component(self, 
                         component_id: str, 
                         input_data: torch.Tensor,
                         callback: Optional[Callable] = None,
                         priority: int = 0) -> str:
        """Execute a component asynchronously."""
        task = ExecutionTask(
            task_id=f"{component_id}_{int(time.time() * 1000)}",
            module_id=component_id,
            input_data=input_data,
            callback=callback,
            priority=priority
        )
        
        return self.task_scheduler.submit_task(task)
    
    def process_batch(self, tasks: List[ExecutionTask]) -> Dict[str, Any]:
        """Process a batch of tasks."""
        # Submit all tasks
        task_ids = []
        for task in tasks:
            task_id = self.task_scheduler.submit_task(task)
            task_ids.append(task_id)
        
        # Execute tasks
        results = self.task_scheduler.execute_tasks()
        
        # Update statistics
        self.execution_stats["total_tasks"] += len(tasks)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # System resources
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Module statistics
        module_stats = {
            "total_modules": len(self.module_manager.modules) + len(self.module_manager.offloaded_modules),
            "active_modules": len(self.module_manager.modules),
            "offloaded_modules": len(self.module_manager.offloaded_modules),
            "memory_usage_mb": self.module_manager._get_current_memory_usage() / (1024 * 1024)
        }
        
        # Task statistics
        task_stats = {
            "pending_tasks": self.task_scheduler.task_queue.qsize(),
            "active_tasks": len(self.task_scheduler.active_tasks),
            "completed_tasks": self.execution_stats["completed_tasks"]
        }
        
        return {
            "system": {
                "memory_percent": memory_info.percent,
                "cpu_percent": cpu_percent,
                "available_memory_gb": memory_info.available / (1024**3)
            },
            "modules": module_stats,
            "tasks": task_stats,
            "execution_stats": self.execution_stats
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage by offloading unused modules."""
        initial_usage = self.module_manager._get_current_memory_usage()
        
        # Get current memory usage
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.config.offload_threshold * 100:
            # Offload least recently used modules
            candidates = [
                (module_id, info) for module_id, info in self.module_manager.module_info.items()
                if info.state == ModuleState.IDLE and module_id in self.module_manager.modules
            ]
            
            candidates.sort(key=lambda x: x[1].last_access)
            
            offloaded_count = 0
            for module_id, _ in candidates[:len(candidates)//2]:  # Offload half
                if self.module_manager.offload_module(module_id):
                    offloaded_count += 1
        
        final_usage = self.module_manager._get_current_memory_usage()
        
        return {
            "initial_memory_mb": initial_usage / (1024 * 1024),
            "final_memory_mb": final_usage / (1024 * 1024),
            "memory_freed_mb": (initial_usage - final_usage) / (1024 * 1024),
            "modules_offloaded": offloaded_count if 'offloaded_count' in locals() else 0
        }


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, config: ExecutionEngineConfig):
        self.config = config
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            # Monitor system resources
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            # Log if needed
            if memory_info.percent > 90:
                print(f"Warning: High memory usage: {memory_info.percent}%")
            
            if cpu_percent > 90:
                print(f"Warning: High CPU usage: {cpu_percent}%")
            
            time.sleep(1)  # Monitor every second
