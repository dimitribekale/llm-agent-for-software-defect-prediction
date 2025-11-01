"""
Git repository tool for extracting commits and code changes.
"""

import os
from typing import Any, List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

try:
    import git
    from git import Repo, GitCommandError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

from ..base import BaseTool, ToolMetadata, ToolExecutionException, ToolValidationException


class CommitData:
    """Structured data for a single commit."""

    def __init__(
        self,
        commit_hash: str,
        author: str,
        author_email: str,
        date: datetime,
        message: str,
        added_lines: List[str],
        removed_lines: List[str],
        changed_files: List[str],
        stats: Dict[str, Any]
    ):
        self.commit_hash = commit_hash
        self.author = author
        self.author_email = author_email
        self.date = date
        self.message = message
        self.added_lines = added_lines
        self.removed_lines = removed_lines
        self.changed_files = changed_files
        self.stats = stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "commit_hash": self.commit_hash,
            "author": self.author,
            "author_email": self.author_email,
            "date": self.date.isoformat() if self.date else None,
            "message": self.message,
            "added_lines": self.added_lines,
            "removed_lines": self.removed_lines,
            "changed_files": self.changed_files,
            "stats": self.stats
        }

    def __repr__(self) -> str:
        return f"CommitData({self.commit_hash[:8]}, files={len(self.changed_files)})"


class GitRepositoryTool(BaseTool):
    """
    Tool for extracting commits and code changes from Git repositories.

    Retrieves commit history, diffs, and metadata for defect prediction analysis.
    """

    def __init__(self):
        super().__init__()
        if not GIT_AVAILABLE:
            raise ImportError(
                "GitPython is required for GitRepositoryTool. "
                "Install it with: pip install GitPython"
            )

    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name="git_repository",
            description="Extract commits and code changes from Git repositories for defect analysis",
            parameters={
                "repo_path": "Path to Git repository (required)",
                "max_commits": "Maximum number of commits to retrieve (default: 100)",
                "branch": "Branch name (default: current branch)",
                "file_extensions": "Filter by file extensions (e.g., ['.py', '.java'])"
            },
            returns="List of CommitData objects with diffs and metadata",
            examples=[
                "repo_path='/path/to/repo', max_commits=50",
                "repo_path='/path/to/repo', branch='main', file_extensions=['.py']"
            ]
        )

    def validate_parameters(self, **kwargs) -> bool:
        """Validate Git repository parameters."""
        repo_path = kwargs.get("repo_path", "")

        if not repo_path or not isinstance(repo_path, str):
            raise ToolValidationException("repo_path must be a non-empty string")

        if not os.path.exists(repo_path):
            raise ToolValidationException(f"Repository path does not exist: {repo_path}")

        if not os.path.isdir(repo_path):
            raise ToolValidationException(f"Repository path is not a directory: {repo_path}")

        # Check if it's a git repository
        git_dir = os.path.join(repo_path, '.git')
        if not os.path.exists(git_dir):
            raise ToolValidationException(
                f"Not a Git repository (no .git directory found): {repo_path}"
            )

        max_commits = kwargs.get("max_commits", 100)
        if not isinstance(max_commits, int) or max_commits < 1:
            raise ToolValidationException("max_commits must be a positive integer")

        return True

    def execute(self, **kwargs) -> List[CommitData]:
        """
        Extract commits from repository.

        Args:
            repo_path: Path to Git repository
            max_commits: Maximum commits to retrieve
            branch: Branch name (optional)
            file_extensions: Filter by file extensions (optional)

        Returns:
            List of CommitData objects
        """
        repo_path = kwargs.get("repo_path")
        max_commits = kwargs.get("max_commits", 100)
        branch = kwargs.get("branch", None)
        file_extensions = kwargs.get("file_extensions", None)

        try:
            # Open repository
            repo = Repo(repo_path)

            # Get commits from specified branch or current
            if branch:
                commits = list(repo.iter_commits(branch, max_count=max_commits))
            else:
                commits = list(repo.iter_commits(max_count=max_commits))

            # Extract commit data with progress bar
            commit_data_list = []
            for commit in tqdm(commits, desc="Extracting commits", unit="commit"):
                commit_data = self._extract_commit_data(
                    commit,
                    repo,
                    file_extensions
                )
                if commit_data:  # Only add if has changes
                    commit_data_list.append(commit_data)

            return commit_data_list

        except GitCommandError as e:
            raise ToolExecutionException(f"Git command failed: {str(e)}")
        except Exception as e:
            raise ToolExecutionException(f"Failed to extract commits: {str(e)}")

    def _extract_commit_data(
        self,
        commit,
        repo: Repo,
        file_extensions: Optional[List[str]] = None
    ) -> Optional[CommitData]:
        """
        Extract data from a single commit.

        Args:
            commit: GitPython commit object
            repo: GitPython repo object
            file_extensions: Filter by file extensions

        Returns:
            CommitData object or None if filtered out
        """
        try:
            # Get commit metadata
            commit_hash = commit.hexsha
            author = commit.author.name
            author_email = commit.author.email
            date = datetime.fromtimestamp(commit.committed_date)
            message = commit.message.strip()

            # Get diff
            if commit.parents:
                # Compare with parent
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
            else:
                # Initial commit - compare with empty tree
                diffs = commit.diff(git.NULL_TREE, create_patch=True)

            # Extract added/removed lines and files
            added_lines = []
            removed_lines = []
            changed_files = []

            total_insertions = 0
            total_deletions = 0

            for diff_item in diffs:
                file_path = diff_item.b_path or diff_item.a_path

                # Filter by file extension if specified
                if file_extensions:
                    if not any(file_path.endswith(ext) for ext in file_extensions):
                        continue

                changed_files.append(file_path)

                # Parse diff content
                if diff_item.diff:
                    diff_text = diff_item.diff.decode('utf-8', errors='ignore')

                    for line in diff_text.split('\n'):
                        if line.startswith('+') and not line.startswith('+++'):
                            added_lines.append(line[1:].strip())
                            total_insertions += 1
                        elif line.startswith('-') and not line.startswith('---'):
                            removed_lines.append(line[1:].strip())
                            total_deletions += 1

            # Skip if no changes (after filtering)
            if not changed_files:
                return None

            # Create stats
            stats = {
                "total_insertions": total_insertions,
                "total_deletions": total_deletions,
                "files_changed": len(changed_files),
                "net_lines": total_insertions - total_deletions
            }

            return CommitData(
                commit_hash=commit_hash,
                author=author,
                author_email=author_email,
                date=date,
                message=message,
                added_lines=added_lines,
                removed_lines=removed_lines,
                changed_files=changed_files,
                stats=stats
            )

        except Exception as e:
            # Log warning but don't fail entire extraction
            print(f"Warning: Failed to extract commit {commit.hexsha[:8]}: {str(e)}")
            return None

    def get_repository_info(self, repo_path: str) -> Dict[str, Any]:
        """
        Get general repository information.

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with repo metadata
        """
        try:
            repo = Repo(repo_path)

            return {
                "path": repo_path,
                "current_branch": repo.active_branch.name,
                "total_commits": len(list(repo.iter_commits())),
                "branches": [b.name for b in repo.branches],
                "remotes": [r.name for r in repo.remotes],
                "is_dirty": repo.is_dirty(),
                "untracked_files": repo.untracked_files
            }
        except Exception as e:
            raise ToolExecutionException(f"Failed to get repository info: {str(e)}")
