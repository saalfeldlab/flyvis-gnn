import os
import subprocess

from .claude_cli import run_claude_cli


def generate_code_brief(memory_path, block_number, case_study, case_study_brief,
                        root_dir, exploration_dir):
    """LLM generates a structured code brief from accumulated knowledge."""
    briefs_dir = os.path.join(exploration_dir, 'briefs')
    os.makedirs(briefs_dir, exist_ok=True)
    brief_path = os.path.join(briefs_dir, f'block_{block_number:03d}_brief.md')

    prompt = f"""Generate a structured code modification brief for block {block_number}.

Read the working memory: {memory_path}

Case study: {case_study}
Case study description: {case_study_brief}

Write a brief to {brief_path} with these sections:
## Code Brief — Block {block_number}
### Context
(Summarize what was learned from previous blocks, current best results)
### Hypothesis
(What structural code change do you propose, and why)
### Required Changes
(List specific files and functions to modify, with rationale)
### Constraints
- Existing configs must still work (backward compatibility)
- Do not break the training/test pipeline
### Success Criteria
(How to verify the code change works)

IMPORTANT: Only write the brief file. Do NOT modify any source code."""

    print(f"\n\033[94m{'='*60}\033[0m")
    print(f"\033[94mGenerating code brief for block {block_number}...\033[0m")
    print(f"\033[94m{'='*60}\033[0m")

    run_claude_cli(prompt, root_dir, max_turns=30,
                   allowed_tools=['Read', 'Write', 'Glob', 'Grep'])

    return brief_path


def interactive_code_session(brief_path, memory_path, analysis_path, root_dir,
                             case_study, cluster_enabled, exploration_dir, block_number):
    """Interactive code modification session within the running script.

    The script stays running. Human interacts via terminal input().
    LLM is called via claude CLI behind the scenes.
    Returns True if code was changed, False if skipped.

    Interaction loop:
      - 'apply'        — apply the current proposal
      - 'alternatives'  — ask LLM to propose 2-3 different approaches
      - 'skip'         — skip code changes entirely
      - any other text — free-form feedback/suggestion to revise the proposal
    After apply, a validation sub-loop allows 'diff', 'revert', feedback, or 'accept'.
    """
    code_tools = ['Read', 'Edit', 'Write', 'Glob', 'Grep']

    context_files = f"""Read the brief: {brief_path}
Read working memory: {memory_path}"""

    # Step 1: LLM generates initial proposal
    proposal_prompt = f"""CODE PROPOSAL SESSION for block {block_number}.

{context_files}

Propose specific code changes. For each file to modify:
- Show the exact function/lines to change
- Explain why
- Show the proposed new code

Do NOT apply changes yet. Just propose and explain."""

    print(f"\n\033[94m{'='*60}\033[0m")
    print(f"\033[94mCODE SESSION: Interactive Code Modification — Block {block_number}\033[0m")
    print(f"\033[94mCase study: {case_study}\033[0m")
    print(f"\033[94m{'='*60}\033[0m\n")

    proposal = run_claude_cli(proposal_prompt, root_dir, max_turns=50,
                              allowed_tools=code_tools)

    # Step 2: Conversation loop — human steers until satisfied
    last_proposal = proposal

    while True:
        print(f"\n\033[93m{'-'*60}\033[0m")
        print("\033[93mOptions:\033[0m")
        print("  'apply'         — apply the current proposal")
        print("  'alternatives'  — ask for 2-3 different approaches to choose from")
        print("  'skip'          — skip code changes, proceed to exploration")
        print("  Or type any feedback / suggestion to revise the proposal")
        print(f"\033[93m{'-'*60}\033[0m")

        human_input = input("\033[96m> \033[0m").strip()

        if not human_input:
            continue

        if human_input.lower() == 'skip':
            print("\033[93mSkipping code changes.\033[0m")
            return False

        if human_input.lower() in ('approve', 'apply'):
            # --- Apply phase ---
            apply_prompt = f"""Apply the code changes from the approved proposal.

{context_files}

Previous proposal (approved by human):
--- PROPOSAL START ---
{last_proposal}
--- PROPOSAL END ---

Now edit the actual source files to implement exactly these changes.
After editing, list every file you changed and summarize each change.

IMPORTANT: Do NOT change config YAML files — only source code (.py files)."""

            print("\n\033[93mApplying code changes...\033[0m")
            run_claude_cli(apply_prompt, root_dir, max_turns=50,
                           allowed_tools=code_tools)

            # --- Validation sub-loop ---
            applied = True
            while applied:
                print(f"\n\033[93m{'-'*60}\033[0m")
                print("\033[93mChanges applied. Options:\033[0m")
                print("  'diff'    — show git diff")
                print("  'accept'  — accept changes and proceed to exploration")
                print("  'revert'  — undo changes, go back to proposal phase")
                print("  Or type feedback for additional modifications")
                print(f"\033[93m{'-'*60}\033[0m")

                val_input = input("\033[96m> \033[0m").strip()

                if not val_input:
                    continue

                if val_input.lower() == 'accept':
                    _record_code_changes(root_dir, analysis_path, block_number,
                                         exploration_dir)
                    return True

                elif val_input.lower() == 'diff':
                    subprocess.run(['git', 'diff', '--stat'], cwd=root_dir)
                    subprocess.run(['git', 'diff'], cwd=root_dir)

                elif val_input.lower() == 'revert':
                    subprocess.run(['git', 'checkout', '--', '.'], cwd=root_dir)
                    print("\033[93mChanges reverted. Back to proposal phase.\033[0m")
                    applied = False  # break out of validation loop

                else:
                    fix_prompt = f"""Human feedback after reviewing applied code changes:
{val_input}

{context_files}

Make the requested modifications to the source files.
After editing, summarize what you changed."""
                    print("\n\033[93mApplying modifications...\033[0m")
                    run_claude_cli(fix_prompt, root_dir, max_turns=30,
                                   allowed_tools=code_tools)

            # If we get here, user reverted — continue outer proposal loop
            continue

        elif human_input.lower() in ('alternatives', 'alt'):
            alt_prompt = f"""The human wants to see alternative approaches.

{context_files}

Your previous proposal was:
--- PROPOSAL START ---
{last_proposal}
--- PROPOSAL END ---

Propose 2-3 DIFFERENT approaches to the same problem. For each:
- Give it a label (Option A, Option B, Option C)
- Summarize the approach in 2-3 sentences
- List the files/functions affected
- Note pros and cons

Do NOT apply any changes. Just present the options clearly."""

            print("\n\033[93mGenerating alternative approaches...\033[0m")
            alternatives = run_claude_cli(alt_prompt, root_dir, max_turns=50,
                                          allowed_tools=code_tools)
            last_proposal = alternatives

        else:
            feedback_prompt = f"""Human feedback on your code proposal:
{human_input}

{context_files}

Your previous proposal was:
--- PROPOSAL START ---
{last_proposal}
--- PROPOSAL END ---

Revise your proposal based on the human's feedback. Show the updated changes.
Do NOT apply changes yet — just show the revised proposal."""

            print("\n\033[93mRevising proposal...\033[0m")
            response = run_claude_cli(feedback_prompt, root_dir, max_turns=30,
                                      allowed_tools=code_tools)
            last_proposal = response


def _record_code_changes(root_dir, analysis_path, block_number, exploration_dir):
    """Record git diff of code changes in the analysis log."""
    try:
        diff_result = subprocess.run(
            ['git', 'diff', '--stat', '--', '.', ':!config/'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )
        diff_stat = diff_result.stdout.strip()

        full_diff = subprocess.run(
            ['git', 'diff', '--', '.', ':!config/'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )

        if diff_stat:
            diff_path = os.path.join(exploration_dir, 'code_diffs',
                                     f'block_{block_number:03d}_code.diff')
            os.makedirs(os.path.dirname(diff_path), exist_ok=True)
            with open(diff_path, 'w') as f:
                f.write(full_diff.stdout)
            print(f"\033[92mCode diff saved: {diff_path}\033[0m")

            if os.path.exists(analysis_path):
                with open(analysis_path, 'a') as f:
                    f.write(f"\n## Block {block_number}: Code Changes (Code Session)\n")
                    f.write(f"```\n{diff_stat}\n```\n\n")

    except Exception as e:
        print(f"\033[91mWarning: could not record code changes: {e}\033[0m")
