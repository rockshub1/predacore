"""
gRPC Service implementation for the Ethical Governance Module (EGM).
"""
import logging
import logging as _logging
import os
from concurrent import futures
from typing import Optional

import grpc
from predacore._vendor.common.logging_utils import log_json
from predacore._vendor.common.protos import (
    egm_pb2,
    egm_pb2_grpc,
)

from .audit_logger import (  # Import concrete logger and abstract base
    AbstractAuditLogger,
    FileAuditLogger,
)
from .rule_engine import BasicRuleEngine  # Import the concrete rule engine


class EthicalGovernanceModuleService(
    egm_pb2_grpc.EthicalGovernanceModuleServiceServicer
):
    """
    Implements the gRPC methods for the Ethical Governance Module service.
    Handles compliance checks, auditing, and sanitization.
    """

    def __init__(
        self,
        rule_engine: "BasicRuleEngine",
        audit_logger: "AbstractAuditLogger",
        logger: logging.Logger | None = None,
    ):
        self.rule_engine = rule_engine
        self.audit_logger = audit_logger
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("EthicalGovernanceModuleService initialized.")

    async def CheckPlanCompliance(
        self,
        request: egm_pb2.CheckPlanComplianceRequest,
        context: grpc.aio.ServicerContext,
    ) -> egm_pb2.ComplianceCheckResultMessage:
        trace_id = None
        try:
            if (
                hasattr(request, "context")
                and request.context
                and "trace_id" in request.context.fields
            ):
                trace_id = request.context.fields["trace_id"].string_value or None
        except Exception:
            trace_id = None
        self.logger.debug(
            f"[trace={trace_id}] Received CheckPlanCompliance request for plan ID: {request.plan.id}"
        )
        try:
            log_json(
                self.logger,
                _logging.INFO,
                "egm.plan_check.start",
                trace_id=trace_id,
                plan_id=str(request.plan.id),
            )
        except Exception:
            pass
        overall_compliant = True
        all_violations = []
        all_warnings = []
        final_justification = "Plan compliant."  # Default justification

        if not request.plan or not request.plan.steps:
            self.logger.warning(
                f"[trace={trace_id}] Received compliance check request for empty plan."
            )
            # Decide if an empty plan is compliant or not - assume compliant for now
            return egm_pb2.ComplianceCheckResultMessage(
                is_compliant=True, justification="Empty plan provided."
            )

        for i, step in enumerate(request.plan.steps):
            self.logger.debug(
                f"[trace={trace_id}] Checking compliance for step {i+1}/{len(request.plan.steps)}: {step.description[:50]}..."
            )
            try:
                # Pass the step message to the rule engine
                # Context could include plan ID, goal ID, user info etc. if needed later
                step_result: egm_pb2.ComplianceCheckResultMessage = (
                    self.rule_engine.check_compliance(step)
                )

                if not step_result.is_compliant:
                    overall_compliant = False
                    all_violations.extend(step_result.violations)
                    all_warnings.extend(step_result.warnings)
                    # Update justification on first failure
                    if final_justification == "Plan compliant.":
                        final_justification = (
                            f"Plan non-compliant due to step {i+1}. See violations."
                        )
                    self.logger.warning(
                        f"[trace={trace_id}] Step {i+1} failed compliance check. Violations: {step_result.violations}"
                    )
                else:
                    # Collect warnings even if compliant
                    all_warnings.extend(step_result.warnings)

            except Exception as e:
                self.logger.error(
                    f"[trace={trace_id}] Error during compliance check for step {i+1} of plan {request.plan.id}: {e}",
                    exc_info=True,
                )
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Error checking compliance for step {i+1}",
                )
                # Return a non-compliant status on error
                return egm_pb2.ComplianceCheckResultMessage(
                    is_compliant=False,
                    justification=f"Internal error during compliance check for step {i+1}.",
                )

        # In log-only mode, allow non-catastrophic violations
        egm_mode = os.getenv("EGM_MODE", "log_only").lower()
        if egm_mode == "log_only":
            # If any HIGH severity violation exists, still fail; else pass but keep violations in result
            has_high = any(
                v.severity == egm_pb2.SeverityLevel.SEVERITY_HIGH
                for v in all_violations
            )
            if not has_high:
                overall_compliant = True
                if all_violations:
                    final_justification = (
                        "Non-catastrophic violations logged (log_only mode)."
                    )

        # Log the overall result before returning
        log_details = {
            "plan_id": request.plan.id,
            "overall_compliant": overall_compliant,
            "num_violations": len(all_violations),
            "num_warnings": len(all_warnings),
        }
        await self.LogEvent(
            request=egm_pb2.LogEventRequest(
                event_type="PLAN_COMPLIANCE_CHECK",
                component="EGM",
                details=egm_pb2.google_dot_protobuf_dot_struct__pb2.Struct(
                    fields={
                        k: egm_pb2.google_dot_protobuf_dot_struct__pb2.Value(
                            string_value=str(v)
                        )
                        if not isinstance(v, bool)
                        else egm_pb2.google_dot_protobuf_dot_struct__pb2.Value(
                            bool_value=v
                        )
                        for k, v in log_details.items()
                    }
                ),  # Basic conversion for logging
                compliance_status=egm_pb2.ComplianceCheckResultMessage(
                    is_compliant=overall_compliant,
                    violations=all_violations,
                    warnings=all_warnings,
                ),
            ),
            context=context,  # Pass context for potential cancellation etc.
        )

        res = egm_pb2.ComplianceCheckResultMessage(
            is_compliant=overall_compliant,
            violations=all_violations,
            warnings=all_warnings,
            justification=final_justification,
        )
        try:
            log_json(
                self.logger,
                _logging.INFO,
                "egm.plan_check.end",
                trace_id=trace_id,
                plan_id=str(request.plan.id),
                is_compliant=bool(res.is_compliant),
            )
        except Exception:
            pass
        return res

    async def CheckActionCompliance(
        self,
        request: egm_pb2.CheckActionComplianceRequest,
        context: grpc.aio.ServicerContext,
    ) -> egm_pb2.ComplianceCheckResultMessage:
        self.logger.debug("Received CheckActionCompliance request")

        # Convert action description and context from Struct if needed
        # For now, assume rule_engine can handle the protobuf Struct directly or we pass dicts
        action_desc_dict = dict(request.action_description)  # Simple conversion
        context_dict = dict(request.context) if request.context.fields else None

        try:
            # Pass the action description and context to the rule engine
            result: egm_pb2.ComplianceCheckResultMessage = (
                self.rule_engine.check_compliance(
                    item_to_check=action_desc_dict,  # Pass dict representation
                    context=context_dict,
                )
            )

            # Log the result
            log_details = {
                "action_description": str(action_desc_dict)[
                    :200
                ],  # Log truncated action
                "context": str(context_dict)[:200],  # Log truncated context
                "is_compliant": result.is_compliant,
                "num_violations": len(result.violations),
                "num_warnings": len(result.warnings),
            }
            await self.LogEvent(
                request=egm_pb2.LogEventRequest(
                    event_type="ACTION_COMPLIANCE_CHECK",
                    component="EGM",  # Or potentially identify the calling component (e.g., Agent ID)
                    details=egm_pb2.google_dot_protobuf_dot_struct__pb2.Struct(
                        fields={
                            k: egm_pb2.google_dot_protobuf_dot_struct__pb2.Value(
                                string_value=str(v)
                            )
                            if not isinstance(v, bool)
                            else egm_pb2.google_dot_protobuf_dot_struct__pb2.Value(
                                bool_value=v
                            )
                            for k, v in log_details.items()
                        }
                    ),
                    compliance_status=result,
                ),
                context=context,
            )

            # Log-only mode: pass actions unless HIGH severity violations present
            egm_mode = os.getenv("EGM_MODE", "log_only").lower()
            final = result
            if egm_mode == "log_only" and not result.is_compliant:
                has_high = any(
                    v.severity == egm_pb2.SeverityLevel.SEVERITY_HIGH
                    for v in result.violations
                )
                if not has_high:
                    final = egm_pb2.ComplianceCheckResultMessage(
                        is_compliant=True,
                        violations=result.violations,
                        warnings=result.warnings,
                        justification="Non-catastrophic violations logged (log_only mode).",
                    )
            if not final.is_compliant:
                self.logger.warning(
                    f"Action failed compliance check. Violations: {final.violations}"
                )
            else:
                self.logger.debug("Action passed compliance check.")
            try:
                log_json(
                    self.logger,
                    _logging.INFO,
                    "egm.action_check.end",
                    is_compliant=bool(final.is_compliant),
                )
            except Exception:
                pass
            return final

        except Exception as e:
            self.logger.error(
                f"Error during action compliance check: {e}", exc_info=True
            )
            await context.abort(
                grpc.StatusCode.INTERNAL, "Error checking action compliance"
            )
            return egm_pb2.ComplianceCheckResultMessage(
                is_compliant=False,
                justification="Internal error during compliance check.",
            )

    async def LogEvent(
        self, request: egm_pb2.LogEventRequest, context: grpc.aio.ServicerContext
    ) -> egm_pb2.LogEventResponse:
        self.logger.debug(
            f"Received LogEvent request: {request.event_type} from {request.component}"
        )
        # TODO: Format log entry and pass to self.audit_logger
        try:
            await self.audit_logger.log(request)  # Call the actual logger
            log_id = (
                "logged"  # Indicate success, actual ID might not be available/needed
            )
            return egm_pb2.LogEventResponse(success=True, log_entry_id=log_id)
        except Exception as e:
            self.logger.error(f"Failed to log event: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to log event")
            return egm_pb2.LogEventResponse(success=False)

    async def SanitizeInput(
        self, request: egm_pb2.SanitizeInputRequest, context: grpc.aio.ServicerContext
    ) -> egm_pb2.SanitizeInputResponse:
        self.logger.debug(
            f"Received SanitizeInput request for context: {request.context}"
        )
        # TODO: Implement basic sanitization logic based on context
        # For v1, might just return the input data unchanged or perform very basic checks
        return egm_pb2.SanitizeInputResponse(sanitized_data=request.input_data)

    async def SanitizeOutput(
        self, request: egm_pb2.SanitizeOutputRequest, context: grpc.aio.ServicerContext
    ) -> egm_pb2.SanitizeOutputResponse:
        self.logger.debug(
            f"Received SanitizeOutput request for context: {request.context}"
        )
        # TODO: Implement basic sanitization logic
        return egm_pb2.SanitizeOutputResponse(sanitized_data=request.output_data)


# Example function to start the server
async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

    # --- Dependency Setup (Replace with actual implementations) ---
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Use BasicRuleEngine
    rule_engine = BasicRuleEngine(logger=logger)

    # Use FileAuditLogger
    # The logger will create the logs directory if needed
    audit_logger = FileAuditLogger(logger=logger)
    # --- End Dependency Setup ---

    egm_pb2_grpc.add_EthicalGovernanceModuleServiceServicer_to_server(
        EthicalGovernanceModuleService(rule_engine, audit_logger, logger), server
    )
    listen_addr = "[::]:50053"  # Assign a port for EGM
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting EthicalGovernanceModuleService on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
