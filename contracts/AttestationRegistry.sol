// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title AttestationRegistry
/// @notice Registro append-only delle attestazioni di report SAST+VEX: collega l'hash del report (SHA-256),
///         il CID IPFS che lo contiene e l'hash di configurazione del modello che lo ha prodotto (Model Provenance).
contract AttestationRegistry {
    struct Attestation {
        address attester;
        uint64 timestamp;
        bytes32 reportHash; // SHA-256 del report JSON (SBOM + VEX)
        bytes32 modelHash; // SHA-256 della configurazione del modello
        string cid; // CID IPFS del report
    }

    Attestation[] private _attestations;
    mapping(bytes32 => uint256) private _idPlusOne; // reportHash => id + 1 (0 = assente)

    event Attested(uint256 indexed id, address indexed attester, bytes32 indexed reportHash, bytes32 modelHash, string cid);

    error EmptyCid();
    error AlreadyAttested(bytes32 reportHash, uint256 id);
    error UnknownAttestation(uint256 id);

    /// @notice Registra un'attestazione. Lo stesso report non può essere attestato due volte.
    function attest(string calldata cid, bytes32 reportHash, bytes32 modelHash) external returns (uint256 id) {
        if (bytes(cid).length == 0) revert EmptyCid();
        uint256 existing = _idPlusOne[reportHash];
        if (existing != 0) revert AlreadyAttested(reportHash, existing - 1);

        id = _attestations.length;
        _attestations.push(
            Attestation({
                attester: msg.sender,
                timestamp: uint64(block.timestamp),
                reportHash: reportHash,
                modelHash: modelHash,
                cid: cid
            })
        );
        _idPlusOne[reportHash] = id + 1;
        emit Attested(id, msg.sender, reportHash, modelHash, cid);
    }

    function count() external view returns (uint256) {
        return _attestations.length;
    }

    function get(uint256 id) external view returns (Attestation memory) {
        if (id >= _attestations.length) revert UnknownAttestation(id);
        return _attestations[id];
    }

    /// @notice Cerca l'attestazione di un report a partire dal suo hash.
    function idOf(bytes32 reportHash) external view returns (bool found, uint256 id) {
        uint256 stored = _idPlusOne[reportHash];
        return (stored != 0, stored == 0 ? 0 : stored - 1);
    }
}
