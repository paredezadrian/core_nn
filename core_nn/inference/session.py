"""
Session Management for CORE-NN.

Handles conversation sessions, memory persistence, and interaction history.
"""

import json
import pickle
import gzip
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from ..config.schema import SessionConfig


@dataclass
class Interaction:
    """Single interaction in a session."""
    timestamp: float
    user_input: str
    model_response: str
    instruction: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SessionMemory:
    """Memory state for a session."""
    bcm_memories: List[Dict[str, Any]]
    igpm_memories: List[Dict[str, Any]]
    rteu_states: Dict[str, Any]
    mlcs_kpacks: List[str]  # List of kpack IDs


class Session:
    """
    Represents a conversation session with CORE-NN.
    
    Manages interaction history, memory state, and session persistence.
    """
    
    def __init__(self, 
                 session_id: str,
                 name: str = "Untitled Session",
                 config: Optional[SessionConfig] = None):
        self.session_id = session_id
        self.name = name
        self.config = config or SessionConfig()
        
        # Session metadata
        self.created_at = time.time()
        self.last_updated = time.time()
        self.interaction_count = 0
        
        # Interaction history
        self.interactions: List[Interaction] = []
        
        # Memory state
        self.memory_state: Optional[SessionMemory] = None
        
        # Session statistics
        self.stats = {
            "total_tokens_generated": 0,
            "total_memory_operations": 0,
            "average_response_time": 0.0,
            "user_satisfaction": None
        }
    
    def add_interaction(self, 
                       user_input: str,
                       model_response: str,
                       instruction: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Add a new interaction to the session."""
        interaction = Interaction(
            timestamp=time.time(),
            user_input=user_input,
            model_response=model_response,
            instruction=instruction,
            metadata=metadata or {}
        )
        
        self.interactions.append(interaction)
        self.interaction_count += 1
        self.last_updated = time.time()
        
        # Update statistics
        if metadata and "response_time" in metadata:
            response_time = metadata["response_time"]
            current_avg = self.stats["average_response_time"]
            self.stats["average_response_time"] = (
                (current_avg * (self.interaction_count - 1) + response_time) / self.interaction_count
            )
        
        if metadata and "tokens_generated" in metadata:
            self.stats["total_tokens_generated"] += metadata["tokens_generated"]
    
    def get_recent_interactions(self, count: int = 5) -> List[Interaction]:
        """Get the most recent interactions."""
        return self.interactions[-count:] if self.interactions else []
    
    def get_interaction_history(self, 
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None) -> List[Interaction]:
        """Get interaction history within time range."""
        if start_time is None and end_time is None:
            return self.interactions.copy()
        
        filtered = []
        for interaction in self.interactions:
            if start_time and interaction.timestamp < start_time:
                continue
            if end_time and interaction.timestamp > end_time:
                continue
            filtered.append(interaction)
        
        return filtered
    
    def search_interactions(self, query: str, limit: int = 10) -> List[Interaction]:
        """Search interactions by content."""
        query_lower = query.lower()
        matches = []
        
        for interaction in self.interactions:
            if (query_lower in interaction.user_input.lower() or 
                query_lower in interaction.model_response.lower()):
                matches.append(interaction)
                
                if len(matches) >= limit:
                    break
        
        return matches
    
    def save_memory_state(self, model_memories: Dict[str, Any]):
        """Save current model memory state."""
        self.memory_state = SessionMemory(
            bcm_memories=model_memories.get("bcm_memories", []),
            igpm_memories=model_memories.get("igpm_memories", []),
            rteu_states=model_memories.get("rteu_states", {}),
            mlcs_kpacks=model_memories.get("mlcs_kpacks", [])
        )
        self.stats["total_memory_operations"] += 1
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        duration = self.last_updated - self.created_at
        
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_updated": datetime.fromtimestamp(self.last_updated).isoformat(),
            "duration_minutes": duration / 60,
            "interaction_count": self.interaction_count,
            "stats": self.stats,
            "has_memory_state": self.memory_state is not None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "interaction_count": self.interaction_count,
            "interactions": [asdict(interaction) for interaction in self.interactions],
            "memory_state": asdict(self.memory_state) if self.memory_state else None,
            "stats": self.stats,
            "config": asdict(self.config)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            name=data["name"],
            config=SessionConfig(**data.get("config", {}))
        )
        
        session.created_at = data["created_at"]
        session.last_updated = data["last_updated"]
        session.interaction_count = data["interaction_count"]
        session.stats = data.get("stats", {})
        
        # Restore interactions
        for interaction_data in data.get("interactions", []):
            interaction = Interaction(**interaction_data)
            session.interactions.append(interaction)
        
        # Restore memory state
        if data.get("memory_state"):
            session.memory_state = SessionMemory(**data["memory_state"])
        
        return session


class SessionManager:
    """
    Manages multiple sessions and handles persistence.
    
    Provides session creation, loading, saving, and cleanup functionality.
    """
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.session_dir = Path(config.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Active sessions
        self.active_sessions: Dict[str, Session] = {}
        
        # Session index
        self.session_index = self._load_session_index()
    
    def create_session(self, name: str = "New Session") -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = Session(session_id, name, self.config)
        
        self.active_sessions[session_id] = session
        self._update_session_index(session)
        
        return session
    
    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        # Check if already active
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from disk
        session_file = self.session_dir / f"{session_id}.session"
        if not session_file.exists():
            return None
        
        try:
            with gzip.open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            session = Session.from_dict(session_data)
            self.active_sessions[session_id] = session
            
            return session
            
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def save_session(self, session: Session) -> bool:
        """Save a session to disk."""
        try:
            session_file = self.session_dir / f"{session.session_id}.session"
            
            with gzip.open(session_file, 'wb') as f:
                pickle.dump(session.to_dict(), f)
            
            self._update_session_index(session)
            return True
            
        except Exception as e:
            print(f"Error saving session {session.session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Remove file
            session_file = self.session_dir / f"{session_id}.session"
            if session_file.exists():
                session_file.unlink()
            
            # Update index
            if session_id in self.session_index:
                del self.session_index[session_id]
                self._save_session_index()
            
            return True
            
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List available sessions."""
        sessions = []
        
        for session_id, session_info in self.session_index.items():
            sessions.append(session_info)
            
            if len(sessions) >= limit:
                break
        
        # Sort by last updated (most recent first)
        sessions.sort(key=lambda x: x["last_updated"], reverse=True)
        
        return sessions
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Clean up old sessions."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0
        
        sessions_to_delete = []
        for session_id, session_info in self.session_index.items():
            if session_info["last_updated"] < cutoff_time:
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            if self.delete_session(session_id):
                deleted_count += 1
        
        return deleted_count
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        total_sessions = len(self.session_index)
        active_sessions = len(self.active_sessions)
        
        total_interactions = sum(
            info["interaction_count"] for info in self.session_index.values()
        )
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_interactions": total_interactions,
            "session_directory": str(self.session_dir),
            "auto_save_enabled": self.config.auto_save
        }
    
    def _load_session_index(self) -> Dict[str, Dict[str, Any]]:
        """Load session index from disk."""
        index_file = self.session_dir / "session_index.json"
        
        if not index_file.exists():
            return {}
        
        try:
            with open(index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session index: {e}")
            return {}
    
    def _save_session_index(self):
        """Save session index to disk."""
        index_file = self.session_dir / "session_index.json"
        
        try:
            with open(index_file, 'w') as f:
                json.dump(self.session_index, f, indent=2)
        except Exception as e:
            print(f"Error saving session index: {e}")
    
    def _update_session_index(self, session: Session):
        """Update session index with session info."""
        self.session_index[session.session_id] = {
            "session_id": session.session_id,
            "name": session.name,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "interaction_count": session.interaction_count
        }
        
        # Limit index size
        if len(self.session_index) > self.config.max_session_history:
            # Remove oldest sessions
            sorted_sessions = sorted(
                self.session_index.items(),
                key=lambda x: x[1]["last_updated"]
            )
            
            for session_id, _ in sorted_sessions[:-self.config.max_session_history]:
                del self.session_index[session_id]
        
        self._save_session_index()
